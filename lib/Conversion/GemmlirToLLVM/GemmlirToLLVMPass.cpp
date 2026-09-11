//===- GemmlirToLLVMPass.cpp - Gemmlir to LLVM ----------*- C++ -*-===//
//
// Lowers `gemmlir.matmul_i8` into a call to an external C runtime function
// (e.g. `tiled_matmul_auto`) and uses standard conversions to lower the
// remainder to the LLVM dialect. Target MLIR: 22.x (opaque pointers).
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gemmlir/GemmlirDialect.h"
#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirPasses.h"

using namespace mlir;
namespace LLVM = mlir::LLVM;

namespace mlir::gemmlir {

#define GEN_PASS_DEF_CONVERTGEMMLIRTOLLVM
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

// gemmlir.matmul_i8 → LLVM call to runtime
class GemmlirMatmulOpToLLVM : public OpConversionPattern<MatMulInt8Op> {
public:
  using OpConversionPattern::OpConversionPattern; // inject LLVMTypeConverter

  LogicalResult matchAndRewrite(MatMulInt8Op op, MatMulInt8Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // LLVM scalar and pointer types (opaque pointer).
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto i8Ty = rewriter.getI8Type();
    auto f32Ty = rewriter.getF32Type();
    auto boolTy = rewriter.getI1Type();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // External runtime signature: use opaque ptr for all pointers.
    SmallVector<Type> params = {
        i64Ty, i64Ty, i64Ty,     // dim_I, dim_J, dim_K
        ptrTy, ptrTy, ptrTy,     // A, B, D(bias/NULL)
        ptrTy,                   // C
        i64Ty, i64Ty, i64Ty, i64Ty, // stride_A, stride_B, stride_D, stride_C
        f32Ty, f32Ty, i32Ty,     // scale * 2, scale_acc
        i32Ty,                   // act
        f32Ty, f32Ty,            // scale_identity, bert_scale
        boolTy, boolTy, boolTy, boolTy, boolTy, // flags
        i8Ty,                    // weightA (placeholder)
        i32Ty                    // tiled_matmul_type
    };
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, params, /*isVarArg=*/false);

    // Create or fetch the runtime symbol.
    StringRef funcName = "tiled_matmul_auto";
    LLVM::LLVMFuncOp funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType,
                                                 LLVM::linkage::Linkage::External);
    }

    // Shapes.
    auto lhsType = llvm::cast<MemRefType>(op.getLhsMat().getType());
    auto rhsType = llvm::cast<MemRefType>(op.getRhsMat().getType());
    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];

    // Access memref base pointers from converted operands.
    MemRefDescriptor descA(adaptor.getLhsMat());
    MemRefDescriptor descB(adaptor.getRhsMat());
    MemRefDescriptor descC(adaptor.getOutMat());
    Value ptrA = descA.alignedPtr(rewriter, loc);
    Value ptrB = descB.alignedPtr(rewriter, loc);
    Value ptrC = descC.alignedPtr(rewriter, loc);

    // Bias: NULL (use zero initializer for opaque pointer)
    Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrTy);

    // Constants and flags.
    Value stride_A = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(K));
    Value stride_B = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(N));
    Value stride_C = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(N));
    Value stride_D = stride_B;

    Value dim_I = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(M));
    Value dim_K = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(K));
    Value dim_J = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(N));

    auto f32_one = rewriter.getF32FloatAttr(1.0f);
    Value scale_identity = rewriter.create<LLVM::ConstantOp>(loc, f32Ty, f32_one);
    Value scale_acc = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value bert_scale = scale_identity;

    Value repeating_bias = rewriter.create<LLVM::ConstantOp>(loc, boolTy, rewriter.getBoolAttr(false));
    Value transpose_A = rewriter.create<LLVM::ConstantOp>(loc, boolTy, op.getTransposeLhsAttr());
    Value transpose_B = rewriter.create<LLVM::ConstantOp>(loc, boolTy, op.getTransposeRhsAttr());
    Value full_C = rewriter.create<LLVM::ConstantOp>(loc, boolTy, rewriter.getBoolAttr(true));
    Value low_D = rewriter.create<LLVM::ConstantOp>(loc, boolTy, rewriter.getBoolAttr(false));

    Value act = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(0));
    Value weightA = rewriter.create<LLVM::ConstantOp>(loc, i8Ty, rewriter.getI8IntegerAttr(0));
    Value tiled_type = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(0));

    // Optional: Inline asm hook (kept as in original project).
    auto asmStr = rewriter.getStringAttr(".insn r 0x7B, 0x3, 7, x0, x0, x0");
    auto constraints = rewriter.getStringAttr("~{memory}");
    auto se = mlir::UnitAttr::get(rewriter.getContext());
    auto as = mlir::UnitAttr::get(rewriter.getContext());
    auto tailNone = LLVM::TailCallKindAttr::get(rewriter.getContext(),
                                               LLVM::tailcallkind::TailCallKind::None);
    auto dialect = LLVM::AsmDialectAttr::get(rewriter.getContext(), LLVM::AsmDialect::AD_ATT);
    rewriter.create<LLVM::InlineAsmOp>(loc, Type(), ValueRange{}, asmStr, constraints,
                                       se, as, tailNone, dialect, /*operand_attrs=*/nullptr);

    // Call runtime.
    rewriter.create<LLVM::CallOp>(loc, funcOp,
        ValueRange{dim_I, dim_J, dim_K, ptrA, ptrB, nullPtr, ptrC,
                   stride_A, stride_B, stride_D, stride_C,
                   scale_identity, scale_identity, scale_acc, act,
                   scale_identity, bert_scale,
                   repeating_bias, transpose_A, transpose_B, full_C, low_D,
                   weightA, tiled_type});

    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertGemmlirToLLVM
    : public impl::ConvertGemmlirToLLVMBase<ConvertGemmlirToLLVM> {
public:
  using impl::ConvertGemmlirToLLVMBase<
      ConvertGemmlirToLLVM>::ConvertGemmlirToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gemmlir::GemmlirDialect, LLVM::LLVMDialect, memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // Type converter and target.
    LLVMTypeConverter converter(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<MatMulInt8Op>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.markUnknownOpDynamicallyLegal([&](Operation *op) { return true; });

    // Patterns.
    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    patterns.add<GemmlirMatmulOpToLLVM>(converter, &getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
