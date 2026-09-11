//===- LinalgToGemmlirPass.cpp - Linalg to Gemmlir --------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gemmlir/GemmlirDialect.h"
#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_CONVERTLINALGTOGEMMLIR
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

// Rewrite linalg.matmul to gemmlir.matmul_* ops depending on output element type.
class LinalgMatmulOpToGemmlir : public ConversionPattern {
public:
  LinalgMatmulOpToGemmlir(MLIRContext *ctx)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), /*benefit=*/1,
                          ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                                ConversionPatternRewriter &rewriter) const final {
    auto matmul = llvm::dyn_cast<linalg::MatmulOp>(op);
    if (!matmul)
      return failure();

    SmallVector<Value, 2> inputs(matmul.getInputs().begin(), matmul.getInputs().end());
    SmallVector<Value, 1> outputs(matmul.getOutputs().begin(), matmul.getOutputs().end());

    if (inputs.size() != 2 || outputs.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected two inputs and one output");

    for (Value v : inputs) {
      auto mt = llvm::dyn_cast<MemRefType>(v.getType());
      if (!mt || !mt.getElementType().isInteger(8))
        return rewriter.notifyMatchFailure(op, "inputs must be memref<...xi8>");
    }

    Value lhs = inputs[0];
    Value rhs = inputs[1];
    Value out = outputs[0];
    auto outType = llvm::dyn_cast<MemRefType>(out.getType());
    if (!outType)
      return rewriter.notifyMatchFailure(op, "output must be a memref");

    if (outType.getElementType().isInteger(32)) {
      rewriter.create<MatMulInt8Op>(op->getLoc(), lhs, rhs, out,
                                    rewriter.getBoolAttr(false),
                                    rewriter.getBoolAttr(false));
    } else if (outType.getElementType().isInteger(8)) {
      rewriter.create<MatMulInt8ScaleOp>(op->getLoc(), lhs, rhs, out,
                                         rewriter.getBoolAttr(false),
                                         rewriter.getBoolAttr(false));
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported output element type");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLinalgToGemmlir
    : public impl::ConvertLinalgToGemmlirBase<ConvertLinalgToGemmlir> {
public:
  using impl::ConvertLinalgToGemmlirBase<
      ConvertLinalgToGemmlir>::ConvertLinalgToGemmlirBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect, gemmlir::GemmlirDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    ConversionTarget target(getContext());
    target.addLegalDialect<gemmlir::GemmlirDialect, linalg::LinalgDialect, memref::MemRefDialect, func::FuncDialect>();
    target.addIllegalOp<linalg::MatmulOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgMatmulOpToGemmlir>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir

