//===- ForceQuantizedMatmulPass.cpp -----------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <limits>

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FORCEQUANTIZEDMATMUL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

  // Rewrites tensor linalg.matmul(f32,f32)->f32 into a quantized pipeline:
  // qcast(A,f32->!quant.uniform<i8:f32,0.02>) -> scast -> i8
  // qcast(B,f32->!quant.uniform<i8:f32,0.02>) -> scast -> i8
  // C = fill 0 : tensor<...xi32>
  // matmul(i8,i8)->i32
  // scast(i32 -> !quant.uniform<i32:f32, 0.02*0.02>) -> dcast -> f32
  class LinalgMatmulOpQuantizer : public OpRewritePattern<linalg::MatmulOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                  PatternRewriter &rewriter) const final {
      Location loc = op.getLoc();

      // Expect exactly 2 inputs and 1 output (tensor form).
      if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
        return rewriter.notifyMatchFailure(op, "expected 2 inputs and 1 output");

      Value lhs = op.getInputs()[0];
      Value rhs = op.getInputs()[1];
      Value outInit = op.getOutputs()[0];

      auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
      auto outTy = dyn_cast<RankedTensorType>(outInit.getType());

      if (!lhsTy || !rhsTy || !outTy)
        return rewriter.notifyMatchFailure(op, "only supports tensor operands/results");

      if (!lhsTy.getElementType().isF32() || !rhsTy.getElementType().isF32() ||
          !outTy.getElementType().isF32())
        return rewriter.notifyMatchFailure(op, "expects f32 element types");

      // Static shapes for simplicity (can be extended to dynamic later).
      if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() || !outTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "requires static shapes");

      // Construct per-tensor symmetric uniform quant types.
      auto f32 = rewriter.getF32Type();
      auto i8 = rewriter.getIntegerType(8);
      auto i32 = rewriter.getI32Type();

      double scaleI8 = 2.0e-2;                // 0.02
      double scaleAcc = scaleI8 * scaleI8;    // 0.0004

      auto qI8 = quant::UniformQuantizedType::get(
          quant::QuantizationFlags::Signed, /*storageType=*/i8,
          /*expressedType=*/f32, /*scale=*/scaleI8, /*zeroPoint=*/0,
          /*storageTypeMin=*/-128, /*storageTypeMax=*/127);

      auto qI32 = quant::UniformQuantizedType::get(
          quant::QuantizationFlags::Signed, /*storageType=*/i32,
          /*expressedType=*/f32, /*scale=*/scaleAcc, /*zeroPoint=*/0,
          /*storageTypeMin=*/std::numeric_limits<int32_t>::min(),
          /*storageTypeMax=*/std::numeric_limits<int32_t>::max());

      auto lhsQTy = RankedTensorType::get(lhsTy.getShape(), qI8);
      auto rhsQTy = RankedTensorType::get(rhsTy.getShape(), qI8);
      auto lhsI8Ty = RankedTensorType::get(lhsTy.getShape(), i8);
      auto rhsI8Ty = RankedTensorType::get(rhsTy.getShape(), i8);
      auto outI32Ty = RankedTensorType::get(outTy.getShape(), i32);
      auto outQI32Ty = RankedTensorType::get(outTy.getShape(), qI32);

      // A_quant/B_quant: quant.qcast f32 -> quantized<i8:f32,0.02>
      Value lhsQ = rewriter.create<quant::QuantizeCastOp>(loc, lhsQTy, lhs);
      Value rhsQ = rewriter.create<quant::QuantizeCastOp>(loc, rhsQTy, rhs);

      // A_i8/B_i8: quant.scast quantized -> storage (i8)
      Value lhsI8 = rewriter.create<quant::StorageCastOp>(loc, lhsI8Ty, lhsQ);
      Value rhsI8 = rewriter.create<quant::StorageCastOp>(loc, rhsI8Ty, rhsQ);

      // C_base: tensor.empty : tensor<...xi32>
      Value c0 = rewriter.create<arith::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(0));
      // Create an empty tensor with the desired i32 result type. In MLIR 22,
      // tensor::EmptyOp builders take either sizes+element type or the result
      // type with dynamic sizes. All static dims -> no dynamic sizes.
      Value cbase = tensor::EmptyOp::create(rewriter, loc, outI32Ty, ValueRange{}).getResult();
      // C = linalg.fill ins(%c0) outs(%cbase) -> tensor<...xi32>
      Value cFilled = rewriter.create<linalg::FillOp>(loc, ValueRange{c0}, ValueRange{cbase})
                             .getResult(0);

      // result_i32 = linalg.matmul ins(lhsI8, rhsI8) outs(cFilled) -> tensor<...xi32>
      auto resultI32 = rewriter
                           .create<linalg::MatmulOp>(loc, TypeRange{outI32Ty},
                                                      ValueRange{lhsI8, rhsI8},
                                                      ValueRange{cFilled})
                           .getResult(0);

      // result_quant = quant.scast i32 -> quantized<i32:f32, 0.0004>
      Value resultQ = rewriter.create<quant::StorageCastOp>(loc, outQI32Ty, resultI32);
      // result = quant.dcast -> f32
      auto outF32Ty = RankedTensorType::get(outTy.getShape(), f32);
      Value resultF32 = rewriter.create<quant::DequantizeCastOp>(loc, outF32Ty, resultQ);

      rewriter.replaceOp(op, resultF32);
      return success();
    }
  };

  // Conversion pass that applies the rewrite greedily.
  class ForceQuantizedMatmul
      : public impl::ForceQuantizedMatmulBase<ForceQuantizedMatmul> {
public:
    using impl::ForceQuantizedMatmulBase<
        ForceQuantizedMatmul>::ForceQuantizedMatmulBase;

    void getDependentDialects(DialectRegistry& registry) const final
    {
      registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                      memref::MemRefDialect, func::FuncDialect,
                      tensor::TensorDialect, quant::QuantDialect>();
    }

    void runOnOperation() final
    {
      ModuleOp module = getOperation();
      RewritePatternSet patterns(&getContext());
      patterns.add<LinalgMatmulOpQuantizer>(&getContext());
      if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
        signalPassFailure();
    }
  };

} // namespace

} // namespace mlir::gemmlir
