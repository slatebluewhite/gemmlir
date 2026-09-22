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

// gemmini_flush(0): custom-3, funct3 = 3, funct7 = k_FLUSH (7), rs1 = rs2 = x0.
// Emitted before every call that actually reaches the accelerator.
static void emitGemminiFlush(ConversionPatternRewriter &rewriter, Location loc) {
  MLIRContext *ctx = rewriter.getContext();
  auto asmStr = rewriter.getStringAttr(".insn r 0x7B, 0x3, 7, x0, x0, x0");
  auto constraints = rewriter.getStringAttr("~{memory}");
  auto se = mlir::UnitAttr::get(ctx);
  auto as = mlir::UnitAttr::get(ctx);
  auto tailNone = LLVM::TailCallKindAttr::get(
      ctx, LLVM::tailcallkind::TailCallKind::None);
  auto dialect = LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
  rewriter.create<LLVM::InlineAsmOp>(loc, Type(), ValueRange{}, asmStr,
                                     constraints, se, as, tailNone, dialect,
                                     /*operand_attrs=*/nullptr);
}

// gemmini.h refuses these combinations outright (it prints and exits), so catch
// them at compile time rather than at run time on the board.
static LogicalResult checkTransposeSupport(Operation *op, bool transposeLhs,
                                           bool transposeRhs, Dataflow dataflow) {
  if (dataflow == Dataflow::OS && (transposeLhs || transposeRhs))
    return op->emitOpError(
        "the 'os' dataflow cannot transpose an operand; use 'ws'");
  if (dataflow == Dataflow::WS && transposeLhs && transposeRhs)
    return op->emitOpError(
        "'ws' can transpose one operand but not both; transpose one of them "
        "ahead of the matmul");
  // And on this board neither one works. `tiled_matmul_auto` takes the flags
  // and the runtime's CPU implementation honours them exactly, but the
  // accelerator does not: measured against a plain-C reference, transposing B
  // came back 2045 of 2048 elements wrong and transposing A 2048 of 2048, the
  // same on every call. Refusing here keeps a silently wrong answer from being
  // compiled; transpose the operand ahead of the matmul instead.
  if (transposeLhs || transposeRhs)
    return op->emitOpError(
        "this board's accelerator does not compute a transposed operand: "
        "measured 2045 of 2048 elements wrong for B and 2048 of 2048 for A, "
        "where the runtime's own CPU implementation was exact. Transpose the "
        "operand ahead of the matmul");
  return success();
}

/// Start of the data a memref denotes: `alignedPtr + offset`. Using alignedPtr
/// alone silently ignores the offset, which is exactly what a `memref.subview`
/// -- a batch slice, say -- carries.
/// These patterns are always built with an LLVMTypeConverter; OpConversionPattern
/// only hands back the generic base.
static const LLVMTypeConverter &llvmConverter(const TypeConverter *converter) {
  return *static_cast<const LLVMTypeConverter *>(converter);
}

static Value dataPtr(ConversionPatternRewriter &rewriter, Location loc,
                     const LLVMTypeConverter &converter, Value converted,
                     Type memrefTy) {
  MemRefDescriptor desc(converted);
  return desc.bufferPtr(rewriter, loc, converter,
                        llvm::cast<MemRefType>(memrefTy));
}

namespace {
/// How the runtime should read an optional bias: its pointer, its row stride,
/// and whether a single row is broadcast down the matrix.
struct BiasOperand {
  Value ptr;
  int64_t strideD;
  bool repeating;
};
} // namespace

// Declares `name` at module scope once, or returns the existing declaration.
static LLVM::LLVMFuncOp getRuntimeFn(ConversionPatternRewriter &rewriter,
                                     Location loc, ModuleOp module,
                                     StringRef name, ArrayRef<Type> params) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;
  auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
  auto funcType = LLVM::LLVMFunctionType::get(voidTy, params, /*isVarArg=*/false);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return rewriter.create<LLVM::LLVMFuncOp>(loc, name, funcType,
                                           LLVM::linkage::Linkage::External);
}

// Gemmini reads through the L2 and does not probe this board's L1, in either
// direction: it reads past stores the host still holds dirty, and the host
// reads lines it has since overwritten. `gemmlir_flush` in the runtime
// displaces the L1. The call goes in unless --place-cache-flushes has shown
// that this particular one cannot matter, so a pipeline that never runs that
// pass keeps every flush.
static void callWithFlushes(ConversionPatternRewriter &rewriter, Location loc,
                            Operation *op, LLVM::LLVMFuncOp fn, ValueRange args) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto flush = [&] {
    LLVM::LLVMFuncOp fl = getRuntimeFn(rewriter, loc, module, "gemmlir_flush", {});
    rewriter.create<LLVM::CallOp>(loc, fl, ValueRange{});
  };
  if (!op->hasAttr("gemmlir.no_flush_before"))
    flush();
  rewriter.create<LLVM::CallOp>(loc, fn, args);
  if (!op->hasAttr("gemmlir.no_flush_after"))
    flush();
}

namespace {
// Everything the two matmul ops need to emit a tiled_matmul_auto call.
struct MatmulLowering {
  ConversionPatternRewriter &rewriter;
  Location loc;
  ModuleOp module;

  Type i64Ty, i32Ty, i8Ty, f32Ty, boolTy, ptrTy;

  MatmulLowering(ConversionPatternRewriter &rewriter, Location loc, ModuleOp module)
      : rewriter(rewriter), loc(loc), module(module),
        i64Ty(rewriter.getI64Type()), i32Ty(rewriter.getI32Type()),
        i8Ty(rewriter.getI8Type()), f32Ty(rewriter.getF32Type()),
        boolTy(rewriter.getI1Type()),
        ptrTy(LLVM::LLVMPointerType::get(rewriter.getContext())) {}

  Value i64(int64_t v) {
    return rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(v));
  }
  Value i32(int32_t v) {
    return rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(v));
  }
  Value f32(float v) {
    return rewriter.create<LLVM::ConstantOp>(loc, f32Ty, rewriter.getF32FloatAttr(v));
  }
  Value f32(FloatAttr v) {
    return rewriter.create<LLVM::ConstantOp>(loc, f32Ty, rewriter.getF32FloatAttr(
        v.getValue().convertToFloat()));
  }
  Value boolean(bool v) {
    return rewriter.create<LLVM::ConstantOp>(loc, boolTy, rewriter.getBoolAttr(v));
  }
  Value boolean(BoolAttr v) {
    return rewriter.create<LLVM::ConstantOp>(loc, boolTy, v);
  }

  LLVM::LLVMFuncOp declare() {
    // The 24-argument tiled_matmul_auto from gemmini.h under the default
    // gemmini_params.h; runtime/gemmlir_rt.c static-asserts these types.
    SmallVector<Type> params = {
        i64Ty, i64Ty, i64Ty,              // dim_I, dim_J, dim_K
        ptrTy, ptrTy, ptrTy, ptrTy,       // A, B, D, C
        i64Ty, i64Ty, i64Ty, i64Ty,       // stride_A, stride_B, stride_D, stride_C
        f32Ty, f32Ty, i32Ty,              // A_scale, B_scale, D_scale
        i32Ty,                            // act
        f32Ty, f32Ty,                     // scale, bert_scale
        boolTy, boolTy, boolTy, boolTy, boolTy, // repeating_bias, trA, trB, full_C, low_D
        i8Ty,                             // weightA
        i32Ty};                           // tiled_matmul_type
    return getRuntimeFn(rewriter, loc, module, "tiled_matmul_auto", params);
  }

  // Materialises the argument list. Kept separate from emitting the call so the
  // caller can put gemmini_flush(0) immediately before the call rather than
  // before a run of constants.
  SmallVector<Value> args(const MatmulShape &shape, Value ptrA, Value ptrB,
                          Value ptrD, Value ptrC, int64_t strideD,
                          bool repeatingBias, Value scaleA, Value scaleB,
                          BoolAttr trA, BoolAttr trB,
                          Value scale, Value bertScale, Act act, bool fullC,
                          Dataflow dataflow) {
    Value one = f32(1.0f);
    // D_scale_factor stays 1: this gemmini_params.h defines MVIN_SCALE_ACC as
    // the identity, so the bias mvin scale would be ignored anyway.
    return {i64(shape.M), i64(shape.N), i64(shape.K), ptrA, ptrB, ptrD, ptrC,
            i64(shape.strideA), i64(shape.strideB), i64(strideD),
            i64(shape.strideC),
            scaleA, scaleB, i32(1),
            i32(static_cast<int32_t>(act)),
            scale, bertScale,
            boolean(repeatingBias), boolean(trA), boolean(trB), boolean(fullC),
            boolean(false),
            rewriter.create<LLVM::ConstantOp>(loc, i8Ty, rewriter.getI8IntegerAttr(0)),
            i32(static_cast<int32_t>(dataflow))};
  }
};
} // namespace

// gemmlir.matmul_i8 -> tiled_matmul_auto with full_C, i32 accumulators out.
class GemmlirMatmulOpToLLVM : public OpConversionPattern<MatMulInt8Op> {
public:
  using OpConversionPattern::OpConversionPattern; // inject LLVMTypeConverter

  LogicalResult matchAndRewrite(MatMulInt8Op op, MatMulInt8Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // This op lowers with full_C = true, because its result memref holds the raw
    // i32 accumulators. gemmini.h's CPU path (matmul_cpu) writes elem_t and has
    // no full_C support, so `cpu` here would quietly store i8 into an i32 buffer.
    if (op.getDataflow() == Dataflow::CPU)
      return op.emitOpError()
             << "dataflow 'cpu' is not available here: this op lowers with "
                "full_C, and gemmini.h's matmul_cpu writes elem_t, so it cannot "
                "produce the i32 accumulators the result memref holds";

    if (failed(checkTransposeSupport(op, op.getTransposeLhs(), op.getTransposeRhs(),
                                     op.getDataflow())))
      return failure();

    MatmulLowering ml(rewriter, loc, op->getParentOfType<ModuleOp>());
    LLVM::LLVMFuncOp fn = ml.declare();

    std::optional<MatmulShape> shape = computeMatmulShape(
        llvm::cast<MemRefType>(op.getLhsMat().getType()),
        llvm::cast<MemRefType>(op.getRhsMat().getType()),
        llvm::cast<MemRefType>(op.getOutMat().getType()),
        op.getTransposeLhs(), op.getTransposeRhs());
    if (!shape)
      return op.emitOpError("operand shapes do not form a matmul");

    const LLVMTypeConverter &conv = llvmConverter(getTypeConverter());
    Value ptrA = dataPtr(rewriter, loc, conv, adaptor.getLhsMat(),
                         op.getLhsMat().getType());
    Value ptrB = dataPtr(rewriter, loc, conv, adaptor.getRhsMat(),
                         op.getRhsMat().getType());
    Value ptrC = dataPtr(rewriter, loc, conv, adaptor.getOutMat(),
                         op.getOutMat().getType());

    // There is one D pointer. C += A*B hands C over as the bias -- verified on
    // hardware that D may alias C, and low_D = false makes the runtime read it
    // as acc_t, matching the i32 element type. An explicit bias takes that slot
    // instead; the verifier has already ruled out asking for both.
    BiasOperand bias{nullptr, shape->strideC, false};
    if (op.getBias()) {
      auto biasTy = llvm::cast<MemRefType>(op.getBias().getType());
      std::optional<int64_t> biasStride = rowStrideOf(biasTy);
      if (!biasStride)
        return op.emitOpError("bias must be row-major with unit-stride columns");
      bias = {dataPtr(rewriter, loc, conv, adaptor.getBias(),
                      op.getBias().getType()),
              *biasStride, biasTy.getShape()[0] == 1};
    } else if (op.getAccumulate()) {
      bias.ptr = ptrC;
    } else {
      bias.ptr = rewriter.create<LLVM::ZeroOp>(loc, ml.ptrTy).getResult();
    }

    SmallVector<Value> args =
        ml.args(*shape, ptrA, ptrB, bias.ptr, ptrC, bias.strideD, bias.repeating,
                ml.f32(op.getLhsScaleAttr()), ml.f32(op.getRhsScaleAttr()),
                op.getTransposeLhsAttr(), op.getTransposeRhsAttr(), ml.f32(1.0f),
                ml.f32(1.0f), Act::NONE, /*fullC=*/true, op.getDataflow());
    emitGemminiFlush(rewriter, loc);
    callWithFlushes(rewriter, loc, op, fn, args);

    rewriter.eraseOp(op);
    return success();
  }
};

// gemmlir.matmul_i8_scale -> tiled_matmul_auto with full_C = false: the
// accumulator is scaled, the activation applied and the result saturated to i8.
class GemmlirMatmulScaleOpToLLVM : public OpConversionPattern<MatMulInt8ScaleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(MatMulInt8ScaleOp op,
                                MatMulInt8ScaleOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (failed(checkTransposeSupport(op, op.getTransposeLhs(), op.getTransposeRhs(),
                                     op.getDataflow())))
      return failure();

    MatmulLowering ml(rewriter, loc, op->getParentOfType<ModuleOp>());
    LLVM::LLVMFuncOp fn = ml.declare();

    std::optional<MatmulShape> shape = computeMatmulShape(
        llvm::cast<MemRefType>(op.getLhsMat().getType()),
        llvm::cast<MemRefType>(op.getRhsMat().getType()),
        llvm::cast<MemRefType>(op.getOutMat().getType()),
        op.getTransposeLhs(), op.getTransposeRhs());
    if (!shape)
      return op.emitOpError("operand shapes do not form a matmul");

    const LLVMTypeConverter &conv = llvmConverter(getTypeConverter());
    Value ptrA = dataPtr(rewriter, loc, conv, adaptor.getLhsMat(),
                         op.getLhsMat().getType());
    Value ptrB = dataPtr(rewriter, loc, conv, adaptor.getRhsMat(),
                         op.getRhsMat().getType());
    Value ptrC = dataPtr(rewriter, loc, conv, adaptor.getOutMat(),
                         op.getOutMat().getType());
    BiasOperand bias{nullptr, shape->strideC, false};
    if (op.getBias()) {
      auto biasTy = llvm::cast<MemRefType>(op.getBias().getType());
      std::optional<int64_t> biasStride = rowStrideOf(biasTy);
      if (!biasStride)
        return op.emitOpError("bias must be row-major with unit-stride columns");
      bias = {dataPtr(rewriter, loc, conv, adaptor.getBias(),
                      op.getBias().getType()),
              *biasStride, biasTy.getShape()[0] == 1};
    } else {
      bias.ptr = rewriter.create<LLVM::ZeroOp>(loc, ml.ptrTy).getResult();
    }

    SmallVector<Value> args =
        ml.args(*shape, ptrA, ptrB, bias.ptr, ptrC, bias.strideD, bias.repeating,
                ml.f32(op.getLhsScaleAttr()), ml.f32(op.getRhsScaleAttr()),
                op.getTransposeLhsAttr(), op.getTransposeRhsAttr(),
                ml.f32(op.getScaleAttr()), ml.f32(op.getBertScaleAttr()),
                op.getAct(), /*fullC=*/false, op.getDataflow());

    // Unlike matmul_i8 this one is expressible on the runtime's CPU path, so
    // `cpu` is allowed and then no accelerator instruction may be emitted.
    if (op.getDataflow() != Dataflow::CPU)
      emitGemminiFlush(rewriter, loc);
    callWithFlushes(rewriter, loc, op, fn, args);

    rewriter.eraseOp(op);
    return success();
  }
};

// gemmlir.resadd_i8 -> tiled_resadd_auto.
class GemmlirResAddOpToLLVM : public OpConversionPattern<ResAddInt8Op> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ResAddInt8Op op, ResAddInt8Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto f32Ty = rewriter.getF32Type();
    auto boolTy = rewriter.getI1Type();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void tiled_resadd_auto(size_t I, size_t J, scale_t A_scale,
    //     scale_t B_scale, acc_scale_t C_scale, const elem_t *A,
    //     const elem_t *B, elem_t *C, bool relu, enum tiled_matmul_type_t)
    SmallVector<Type> params = {i64Ty, i64Ty, f32Ty, f32Ty, f32Ty,
                                ptrTy, ptrTy, ptrTy, boolTy, i32Ty};
    LLVM::LLVMFuncOp fn =
        getRuntimeFn(rewriter, loc, module, "tiled_resadd_auto", params);

    auto outType = llvm::cast<MemRefType>(op.getOutMat().getType());
    int64_t I = outType.getShape()[0];
    int64_t J = outType.getShape()[1];

    // Tried, and it moves nothing. `tiled_resadd_auto` takes two extents and no
    // strides, so `I x J` is only a split of a contiguous run and the operation
    // is elementwise: any split with the same product is the same answer. The
    // splits are not equally fast in isolation -- the array is 16 wide, and the
    // same elements as 1xN against (N/16)x16 measure, in ms, 192: 0.028/0.011,
    // 384: 0.048/0.013, 1536: 0.204/0.027, 12288: 1.644/0.170.
    //
    // In a model it is worth nothing. The LSTM, whose 32 gate sums are 1x192,
    // goes 17.66 -> 17.59; ResNet-18 67.5 -> 67.6 and MobileNetV2 93.8 -> 93.6,
    // both inside the noise. Four models' objects change for it. Not shipped.

    auto konst = [&](Type t, Attribute a) {
      return rewriter.create<LLVM::ConstantOp>(loc, t, llvm::cast<TypedAttr>(a));
    };
    auto f32Of = [&](FloatAttr a) {
      return rewriter.create<LLVM::ConstantOp>(
          loc, f32Ty, rewriter.getF32FloatAttr(a.getValue().convertToFloat()));
    };

    SmallVector<Value> args = {
        konst(i64Ty, rewriter.getI64IntegerAttr(I)),
        konst(i64Ty, rewriter.getI64IntegerAttr(J)),
        f32Of(op.getLhsScaleAttr()), f32Of(op.getRhsScaleAttr()),
        f32Of(op.getOutScaleAttr()),
        MemRefDescriptor(adaptor.getLhsMat()).alignedPtr(rewriter, loc),
        MemRefDescriptor(adaptor.getRhsMat()).alignedPtr(rewriter, loc),
        MemRefDescriptor(adaptor.getOutMat()).alignedPtr(rewriter, loc),
        konst(boolTy, rewriter.getBoolAttr(op.getAct() == Act::RELU)),
        konst(i32Ty, rewriter.getI32IntegerAttr(
                         static_cast<int32_t>(op.getDataflow())))};

    if (op.getDataflow() != Dataflow::CPU)
      emitGemminiFlush(rewriter, loc);
    callWithFlushes(rewriter, loc, op, fn, args);

    rewriter.eraseOp(op);
    return success();
  }
};


// gemmlir.norm_i8 -> tiled_norm_auto.
class GemmlirNormOpToLLVM : public OpConversionPattern<NormInt8Op> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(NormInt8Op op, NormInt8Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto f32Ty = rewriter.getF32Type();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void tiled_norm_auto(size_t I, size_t J, const acc_t *in, elem_t *out,
    //     acc_scale_t C_scale, int act, enum tiled_matmul_type_t)
    SmallVector<Type> params = {i64Ty, i64Ty, ptrTy, ptrTy, f32Ty, i32Ty, i32Ty};
    LLVM::LLVMFuncOp fn =
        getRuntimeFn(rewriter, loc, module, "tiled_norm_auto", params);

    auto inType = llvm::cast<MemRefType>(op.getInput().getType());
    auto konst = [&](Type t, Attribute a) {
      return rewriter.create<LLVM::ConstantOp>(loc, t, llvm::cast<TypedAttr>(a));
    };

    // `tiled_norm_auto` refuses a type of 0 outright ("Unsupported type"), and
    // `tiled_norm` has no CPU path anyway -- it issues the instructions
    // whatever it is handed. WS is the only thing to say.
    SmallVector<Value> args = {
        konst(i64Ty, rewriter.getI64IntegerAttr(inType.getShape()[0])),
        konst(i64Ty, rewriter.getI64IntegerAttr(inType.getShape()[1])),
        MemRefDescriptor(adaptor.getInput()).alignedPtr(rewriter, loc),
        MemRefDescriptor(adaptor.getOutput()).alignedPtr(rewriter, loc),
        rewriter.create<LLVM::ConstantOp>(
            loc, f32Ty,
            rewriter.getF32FloatAttr(op.getScale().convertToFloat())),
        konst(i32Ty, rewriter.getI32IntegerAttr(
                         static_cast<int32_t>(op.getAct()))),
        konst(i32Ty, rewriter.getI32IntegerAttr(
                         static_cast<int32_t>(Dataflow::WS)))};

    emitGemminiFlush(rewriter, loc);
    callWithFlushes(rewriter, loc, op, fn, args);

    rewriter.eraseOp(op);
    return success();
  }
};

/// `gemmlir.memset` is one `memset`. The buffer is contiguous by the op's own
/// verifier, so the byte count is the whole of it.
struct GemmlirMemsetOpToLLVM : public OpConversionPattern<MemsetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MemsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto ty = llvm::cast<MemRefType>(op.getBuffer().getType());
    int64_t elems = 1;
    for (int64_t d : ty.getShape())
      elems *= d;
    int64_t bytes = elems * (ty.getElementType().getIntOrFloatBitWidth() / 8);

    // void gemmlir_memset(void *p, int value, size_t bytes)
    LLVM::LLVMFuncOp fn = getRuntimeFn(rewriter, loc, module, "gemmlir_memset",
                                       {ptrTy, i32Ty, i64Ty});
    SmallVector<Value> args = {
        dataPtr(rewriter, loc, llvmConverter(getTypeConverter()),
                adaptor.getBuffer(), ty),
        rewriter.create<LLVM::ConstantOp>(
            loc, i32Ty, rewriter.getI32IntegerAttr(op.getValue())),
        rewriter.create<LLVM::ConstantOp>(loc, i64Ty,
                                          rewriter.getI64IntegerAttr(bytes))};
    rewriter.create<LLVM::CallOp>(loc, fn, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// Shared by the two convolution patterns: the runtime takes everything as `int`,
// so the shape arithmetic is done here and handed over as i32 constants.
namespace {
struct ConvLowering {
  ConversionPatternRewriter &rewriter;
  Location loc;

  Type i32Ty, f32Ty, boolTy, ptrTy;

  ConvLowering(ConversionPatternRewriter &rewriter, Location loc)
      : rewriter(rewriter), loc(loc), i32Ty(rewriter.getI32Type()),
        f32Ty(rewriter.getF32Type()), boolTy(rewriter.getI1Type()),
        ptrTy(LLVM::LLVMPointerType::get(rewriter.getContext())) {}

  Value i32(int64_t v) {
    return rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(static_cast<int32_t>(v)));
  }
  Value f32(FloatAttr v) {
    return rewriter.create<LLVM::ConstantOp>(
        loc, f32Ty, rewriter.getF32FloatAttr(v.getValue().convertToFloat()));
  }
  Value boolean(bool v) {
    return rewriter.create<LLVM::ConstantOp>(loc, boolTy, rewriter.getBoolAttr(v));
  }
  Value nullPtr() { return rewriter.create<LLVM::ZeroOp>(loc, ptrTy); }
  Value ptrOf(const LLVMTypeConverter &converter, Value converted, Type memrefTy) {
    return dataPtr(rewriter, loc, converter, converted, memrefTy);
  }

  // The convolution's own output extent, i.e. what the runtime calls
  // out_row_dim / out_col_dim -- before any fused pooling.
  static int64_t convOut(int64_t in, int64_t kernel, int64_t stride,
                         int64_t padding, int64_t dilation) {
    return (in + 2 * padding - (dilation * (kernel - 1) + 1)) / stride + 1;
  }
};
} // namespace

// gemmlir.conv2d_i8 -> tiled_conv_auto.
class GemmlirConv2DOpToLLVM : public OpConversionPattern<Conv2DInt8Op> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(Conv2DInt8Op op, Conv2DInt8Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ConvLowering cl(rewriter, loc);

    auto in = llvm::cast<MemRefType>(op.getInput().getType());
    auto flt = llvm::cast<MemRefType>(op.getFilter().getType());

    int64_t batch = in.getShape()[0];
    int64_t inRows = in.getShape()[1], inCols = in.getShape()[2];
    int64_t inCh = flt.getShape()[2], outCh = flt.getShape()[3];
    int64_t kernel = flt.getShape()[0];
    int64_t stride = op.getStride(), padding = op.getPadding();
    int64_t dilation = op.getDilation();
    // The input is read as if `input_dilation - 1` zeros sat between every pair
    // of samples, which is what makes this a transposed convolution. The
    // runtime works the dilated extent out itself and wants the real one, but
    // the output extent is the dilated input's.
    int64_t inputDilation = op.getInputDilation();
    int64_t spreadRows = inputDilation * (inRows - 1) + 1;
    int64_t spreadCols = inputDilation * (inCols - 1) + 1;
    int64_t outRows = ConvLowering::convOut(spreadRows, kernel, stride, padding, dilation);
    int64_t outCols = ConvLowering::convOut(spreadCols, kernel, stride, padding, dilation);

    // How far apart two pixels are. For an ordinary buffer that is the channel
    // count; for one branch's slice of a concatenation it is the joined width,
    // and the convolution writes straight into it rather than anyone copying
    // afterwards.
    auto pixelStride = [](MemRefType t, int64_t channels) -> int64_t {
      int64_t offset = 0;
      SmallVector<int64_t> strides;
      if (failed(t.getStridesAndOffset(strides, offset)) || strides.size() != 4 ||
          strides[3] != 1)
        return -1;
      return strides[2];
    };
    int64_t inStride = pixelStride(in, inCh);
    int64_t outStride =
        pixelStride(llvm::cast<MemRefType>(op.getOutput().getType()), outCh);
    if (inStride < 0 || outStride < 0)
      return op.emitOpError("input and output must have unit-stride channels");

    // int x 15, bool x 5, ptr x 4, int, float, int x 3, int
    SmallVector<Type> params = {
        cl.i32Ty, cl.i32Ty, cl.i32Ty, cl.i32Ty, cl.i32Ty, cl.i32Ty, cl.i32Ty,
        cl.i32Ty, cl.i32Ty, cl.i32Ty, cl.i32Ty, cl.i32Ty,
        cl.i32Ty, cl.i32Ty, cl.i32Ty,
        cl.boolTy, cl.boolTy, cl.boolTy, cl.boolTy, cl.boolTy,
        cl.ptrTy, cl.ptrTy, cl.ptrTy, cl.ptrTy,
        cl.i32Ty, cl.f32Ty,
        cl.i32Ty, cl.i32Ty, cl.i32Ty,
        cl.i32Ty};
    LLVM::LLVMFuncOp fn = getRuntimeFn(rewriter, loc, op->getParentOfType<ModuleOp>(),
                                       "tiled_conv_stride_auto", params);

    SmallVector<Value> args = {
        cl.i32(batch), cl.i32(inRows), cl.i32(inCols), cl.i32(inCh),
        cl.i32(outCh), cl.i32(outRows), cl.i32(outCols),
        cl.i32(stride), cl.i32(inputDilation), cl.i32(dilation),
        cl.i32(padding), cl.i32(kernel),
        cl.i32(inStride), cl.i32(outCh), cl.i32(outStride),
        cl.boolean(false), cl.boolean(false), cl.boolean(false),
        cl.boolean(false), cl.boolean(false),
        cl.ptrOf(llvmConverter(getTypeConverter()), adaptor.getInput(), op.getInput().getType()), cl.ptrOf(llvmConverter(getTypeConverter()), adaptor.getFilter(), op.getFilter().getType()),
        adaptor.getBias() ? cl.ptrOf(llvmConverter(getTypeConverter()), adaptor.getBias(), op.getBias().getType()) : cl.nullPtr(),
        cl.ptrOf(llvmConverter(getTypeConverter()), adaptor.getOutput(), op.getOutput().getType()),
        cl.i32(static_cast<int32_t>(op.getAct())), cl.f32(op.getScaleAttr()),
        cl.i32(op.getPoolSize()), cl.i32(op.getPoolStride()),
        cl.i32(op.getPoolPadding()),
        cl.i32(static_cast<int32_t>(op.getDataflow()))};

    if (op.getDataflow() != Dataflow::CPU)
      emitGemminiFlush(rewriter, loc);
    callWithFlushes(rewriter, loc, op, fn, args);

    rewriter.eraseOp(op);
    return success();
  }
};

// gemmlir.depthwise_conv2d_i8 -> tiled_conv_dw_auto.
class GemmlirDepthwiseConv2DOpToLLVM
    : public OpConversionPattern<DepthwiseConv2DInt8Op> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(DepthwiseConv2DInt8Op op,
                                DepthwiseConv2DInt8Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ConvLowering cl(rewriter, loc);

    auto in = llvm::cast<MemRefType>(op.getInput().getType());
    auto flt = llvm::cast<MemRefType>(op.getFilter().getType());

    int64_t batch = in.getShape()[0];
    int64_t inRows = in.getShape()[1], inCols = in.getShape()[2];
    int64_t channels = flt.getShape()[0], kernel = flt.getShape()[1];
    int64_t stride = op.getStride(), padding = op.getPadding();
    int64_t outRows = ConvLowering::convOut(inRows, kernel, stride, padding, 1);
    int64_t outCols = ConvLowering::convOut(inCols, kernel, stride, padding, 1);

    SmallVector<Type> params = {
        cl.i32Ty, cl.i32Ty, cl.i32Ty, cl.i32Ty, cl.i32Ty, cl.i32Ty,
        cl.i32Ty, cl.i32Ty, cl.i32Ty,
        cl.ptrTy, cl.ptrTy, cl.ptrTy, cl.ptrTy,
        cl.i32Ty, cl.f32Ty,
        cl.i32Ty, cl.i32Ty, cl.i32Ty,
        cl.i32Ty};
    LLVM::LLVMFuncOp fn = getRuntimeFn(rewriter, loc, op->getParentOfType<ModuleOp>(),
                                       "tiled_conv_dw_auto", params);

    SmallVector<Value> args = {
        cl.i32(batch), cl.i32(inRows), cl.i32(inCols), cl.i32(channels),
        cl.i32(outRows), cl.i32(outCols),
        cl.i32(stride), cl.i32(padding), cl.i32(kernel),
        cl.ptrOf(llvmConverter(getTypeConverter()), adaptor.getInput(), op.getInput().getType()), cl.ptrOf(llvmConverter(getTypeConverter()), adaptor.getFilter(), op.getFilter().getType()),
        adaptor.getBias() ? cl.ptrOf(llvmConverter(getTypeConverter()), adaptor.getBias(), op.getBias().getType()) : cl.nullPtr(),
        cl.ptrOf(llvmConverter(getTypeConverter()), adaptor.getOutput(), op.getOutput().getType()),
        cl.i32(static_cast<int32_t>(op.getAct())), cl.f32(op.getScaleAttr()),
        cl.i32(op.getPoolSize()), cl.i32(op.getPoolStride()),
        cl.i32(op.getPoolPadding()),
        cl.i32(static_cast<int32_t>(op.getDataflow()))};

    if (op.getDataflow() != Dataflow::CPU)
      emitGemminiFlush(rewriter, loc);
    callWithFlushes(rewriter, loc, op, fn, args);

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
    target.addIllegalOp<MatMulInt8Op, MatMulInt8ScaleOp, ResAddInt8Op, NormInt8Op,
                        Conv2DInt8Op, DepthwiseConv2DInt8Op, MemsetOp>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.markUnknownOpDynamicallyLegal([&](Operation *op) { return true; });

    // Patterns.
    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    patterns.add<GemmlirMatmulOpToLLVM, GemmlirMatmulScaleOpToLLVM,
                 GemmlirResAddOpToLLVM, GemmlirNormOpToLLVM, GemmlirConv2DOpToLLVM,
                 GemmlirDepthwiseConv2DOpToLLVM, GemmlirMemsetOpToLLVM>(
        converter, &getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
