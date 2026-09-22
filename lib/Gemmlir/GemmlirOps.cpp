//===- GemmlirOps.cpp - Gemmlir dialect ops ---------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirDialect.h"

#define GET_OP_CLASSES
#include "Gemmlir/GemmlirOps.cpp.inc"

using namespace mlir;
using namespace mlir::gemmlir;

/// The runtime addresses a matrix as `base + row*stride + col`, so each operand
/// has to be row-major with unit-stride columns. Returns its row stride, or
/// nullopt when the layout is something else (a column slice, a dynamic stride).
std::optional<int64_t> mlir::gemmlir::rowStrideOf(MemRefType type) {
  auto [strides, offset] = type.getStridesAndOffset();
  if (strides.size() != 2)
    return std::nullopt;
  if (strides[1] != 1 || ShapedType::isDynamic(strides[0]))
    return std::nullopt;
  return strides[0];
}

std::optional<MatmulShape>
mlir::gemmlir::computeMatmulShape(MemRefType lhs, MemRefType rhs, MemRefType out,
                                  bool transposeLhs, bool transposeRhs) {
  if (!lhs.hasStaticShape() || !rhs.hasStaticShape() || !out.hasStaticShape())
    return std::nullopt;

  MatmulShape s;
  // A is stored (M, K), or (K, M) when transposed.
  s.M = transposeLhs ? lhs.getShape()[1] : lhs.getShape()[0];
  int64_t kFromA = transposeLhs ? lhs.getShape()[0] : lhs.getShape()[1];
  // B is stored (K, N), or (N, K) when transposed.
  s.N = transposeRhs ? rhs.getShape()[0] : rhs.getShape()[1];
  int64_t kFromB = transposeRhs ? rhs.getShape()[1] : rhs.getShape()[0];

  if (kFromA != kFromB)
    return std::nullopt;
  s.K = kFromA;
  if (out.getShape()[0] != s.M || out.getShape()[1] != s.N)
    return std::nullopt;

  // Strides come from the layout, not the shape: a slice of a bigger buffer has
  // the same shape but a wider row stride.
  std::optional<int64_t> sa = rowStrideOf(lhs), sb = rowStrideOf(rhs),
                         sc = rowStrideOf(out);
  if (!sa || !sb || !sc)
    return std::nullopt;
  s.strideA = *sa;
  s.strideB = *sb;
  s.strideC = *sc;
  return s;
}

/// Shared shape check for the two matmul ops.
static LogicalResult verifyMatmulShapes(Operation *op, Value lhs, Value rhs,
                                        Value out, Value bias, bool transposeLhs,
                                        bool transposeRhs) {
  auto lhsTy = llvm::cast<MemRefType>(lhs.getType());
  auto rhsTy = llvm::cast<MemRefType>(rhs.getType());
  auto outTy = llvm::cast<MemRefType>(out.getType());
  if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() || !outTy.hasStaticShape())
    return op->emitOpError("operands must have a static shape");

  std::optional<MatmulShape> shape =
      computeMatmulShape(lhsTy, rhsTy, outTy, transposeLhs, transposeRhs);
  if (!shape)
    return op->emitOpError()
           << "operand shapes do not form a matmul: " << lhsTy
           << (transposeLhs ? " (transposed)" : "") << " x " << rhsTy
           << (transposeRhs ? " (transposed)" : "") << " -> " << outTy;

  if (bias) {
    auto biasTy = llvm::cast<MemRefType>(bias.getType());
    if (!biasTy.hasStaticShape())
      return op->emitOpError("bias must have a static shape");
    // The runtime reads D[bias_row * stride_D + j] with bias_row pinned to 0
    // when repeating_bias is set, so one row means "broadcast down the matrix".
    int64_t rows = biasTy.getShape()[0];
    if ((rows != shape->M && rows != 1) || biasTy.getShape()[1] != shape->N)
      return op->emitOpError()
             << "bias must be " << shape->M << "x" << shape->N << " or 1x"
             << shape->N << ", got " << biasTy;
  }
  return success();
}

LogicalResult MatMulInt8Op::verify() {
  // There is one D pointer: it is either the output being accumulated into or
  // a separate bias, never both.
  if (getBias() && getAccumulate())
    return emitOpError("bias and accumulate cannot both be set: the runtime has "
                       "a single bias operand, which accumulate already uses "
                       "for the output. Set accumulate = false, or add the bias "
                       "into the output first");
  return verifyMatmulShapes(*this, getLhsMat(), getRhsMat(), getOutMat(),
                            getBias(), getTransposeLhs(), getTransposeRhs());
}

/// Softmax's output scale is built in: the hardware divides each row by its own
/// sum and multiplies by 127, and the operation's `scale` multiplies on top of
/// that. The runtime's own CPU reference does not -- `matmul_cpu` substitutes
/// `127 / sum_exp` for the scale it was handed and ignores it -- so the two
/// agree only at 1. Measured: with `scale` 0.01 the accelerator returned
/// exactly one hundredth of the reference, element for element, and at 1.0 the
/// two are bit-identical. Anything else is a silently different answer, so it
/// is refused; put a requantization after the softmax instead.
static LogicalResult verifySoftmaxScale(Operation *op, Act act,
                                        llvm::APFloat scale) {
  if (act != Act::SOFTMAX || scale.convertToFloat() == 1.0f)
    return success();
  return op->emitOpError(
      "softmax already scales by 127 over the row's sum, and the runtime's CPU "
      "reference ignores any other scale; use 1.0 and requantize afterwards");
}

LogicalResult MatMulInt8ScaleOp::verify() {
  if (failed(verifySoftmaxScale(*this, getAct(), getScale())))
    return failure();
  return verifyMatmulShapes(*this, getLhsMat(), getRhsMat(), getOutMat(),
                            getBias(), getTransposeLhs(), getTransposeRhs());
}

LogicalResult ResAddInt8Op::verify() {
  auto lhs = llvm::cast<MemRefType>(getLhsMat().getType());
  auto rhs = llvm::cast<MemRefType>(getRhsMat().getType());
  auto out = llvm::cast<MemRefType>(getOutMat().getType());

  // tiled_resadd_auto takes one stride for all three operands and derives it
  // from J, so the shapes have to be identical, not merely the same size.
  if (lhs.getShape() != rhs.getShape() || lhs.getShape() != out.getShape())
    return emitOpError("operands must all have the same shape, got ")
           << lhs << ", " << rhs << " and " << out;
  if (!lhs.hasStaticShape())
    return emitOpError("operands must have a static shape");
  return success();
}

/// Extent of one convolution output axis, before any fused pooling.
static int64_t convOutExtent(int64_t in, int64_t kernel, int64_t stride,
                             int64_t padding, int64_t dilation) {
  int64_t effective = dilation * (kernel - 1) + 1;
  return (in + 2 * padding - effective) / stride + 1;
}

/// Extent after the fused max-pool, or `convOut` when pooling is off.
static int64_t poolOutExtent(int64_t convOut, int64_t poolSize,
                             int64_t poolStride, int64_t poolPadding) {
  if (poolStride == 0)
    return convOut;
  return (convOut + 2 * poolPadding - poolSize) / poolStride + 1;
}

/// Shared shape checking for the two convolution ops. `channels` is the input
/// channel count and `outChannels` what the filter produces.
/// The runtime walks an NHWC buffer as `((n * rows + r) * cols + c) * stride`,
/// with `stride` the one number it takes between two pixels. A window that is
/// narrower than the buffer it sits in has a wider row than that -- and the
/// runtime cannot be told, so it reads and writes the wrong rows and nothing
/// says so. A convolution writing the middle of a padded buffer is exactly that
/// window: measured, 0.7742 relative L2 for a 1x1 convolution feeding a grouped
/// one, where the same pair with no padded destination is 0.0126.
///
/// A *channel* window is fine and is what a concatenation makes: the pixel
/// stride widens and the rows widen with it.
static bool rowsFollowTheStride(MemRefType t) {
  int64_t offset = 0;
  SmallVector<int64_t> strides;
  if (failed(t.getStridesAndOffset(strides, offset)) || strides.size() != 4)
    return false;
  if (strides[3] != 1)
    return false;
  return strides[1] == t.getShape()[2] * strides[2] &&
         strides[0] == t.getShape()[1] * strides[1];
}

static LogicalResult verifyConvShapes(Operation *op, MemRefType input,
                                      MemRefType output, Value bias,
                                      int64_t channels, int64_t outChannels,
                                      int64_t kernel, int64_t stride,
                                      int64_t padding, int64_t dilation,
                                      int64_t poolSize, int64_t poolStride,
                                      int64_t poolPadding,
                                      int64_t inputDilation = 1) {
  auto err = [&]() { return op->emitOpError(); };

  if (!input.hasStaticShape() || !output.hasStaticShape())
    return err() << "input and output must have a static shape";
  if (!rowsFollowTheStride(input) || !rowsFollowTheStride(output))
    return err() << "input and output rows must be as wide as their pixel "
                    "stride says: the runtime takes one stride between pixels "
                    "and derives the row from it";
  if (stride < 1 || dilation < 1 || padding < 0)
    return err() << "stride and dilation must be positive and padding non-negative";
  // `tiled_conv_auto` refuses a padding that reaches the kernel:
  // `if (kernel_dim <= padding) { printf("kernel_dim must be larger than
  // padding\n"); exit(1); }`. It compares against the *undilated* kernel, so a
  // dilated convolution's shape-preserving padding -- `dilation*(K-1)/2`, which
  // is 4 for a 3-tap filter at rate 4 -- is past the limit even though the
  // dilated filter is 9 taps wide. Such a padding has to be materialized;
  // `FoldPaddingIntoConv` leaves it alone rather than folding it in here.
  if (padding >= kernel)
    return err() << "padding " << padding << " must be smaller than the "
                 << kernel << "-tap kernel: the runtime compares against the "
                    "undilated kernel_dim and exits";
  // The runtime's accelerator path takes 1 or 2 and only with a unit stride;
  // anything else reaches a `printf` and an `exit(1)` inside gemmini.h.
  if (inputDilation < 1 || inputDilation > 2)
    return err() << "input_dilation must be 1 or 2, got " << inputDilation;
  if (inputDilation > 1 && stride != 1)
    return err() << "input_dilation is only available with a unit stride";
  if (poolStride < 0 || poolSize < 0 || poolPadding < 0)
    return err() << "pooling parameters must be non-negative";
  if (poolStride != 0 && poolSize < 1)
    return err() << "pool_size must be positive when pooling is enabled";

  if (input.getShape()[3] != channels)
    return err() << "input has " << input.getShape()[3]
                 << " channels but the filter expects " << channels;
  if (output.getShape()[0] != input.getShape()[0])
    return err() << "input and output batch sizes differ";
  if (output.getShape()[3] != outChannels)
    return err() << "output has " << output.getShape()[3]
                 << " channels but the filter produces " << outChannels;

  for (int axis = 0; axis < 2; axis++) {
    // Dilating the input spreads it: the convolution sees
    // `input_dilation * (in - 1) + 1` samples.
    int64_t in = inputDilation * (input.getShape()[axis + 1] - 1) + 1;
    int64_t convOut = convOutExtent(in, kernel, stride, padding, dilation);
    if (convOut < 1)
      return err() << "input extent " << in << " on axis " << axis
                   << " is too small for a " << kernel << "-tap kernel";
    int64_t want = poolOutExtent(convOut, poolSize, poolStride, poolPadding);
    if (output.getShape()[axis + 1] != want)
      return err() << "output extent on axis " << axis << " is "
                   << output.getShape()[axis + 1] << ", expected " << want;
  }

  if (bias) {
    auto biasType = llvm::cast<MemRefType>(bias.getType());
    if (!biasType.hasStaticShape() || biasType.getShape()[0] != outChannels)
      return err() << "bias must hold one value per output channel ("
                   << outChannels << ")";
  }
  return success();
}

LogicalResult NormInt8Op::verify() {
  auto in = llvm::cast<MemRefType>(getInput().getType());
  auto out = llvm::cast<MemRefType>(getOutput().getType());
  if (!in.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("input and output must have a static shape");
  if (in.getShape() != out.getShape())
    return emitOpError("input and output must have the same shape, got ")
           << in << " and " << out;
  // The runtime walks both with `J` as the row stride, so a row has to be the
  // row it thinks it is.
  for (MemRefType t : {in, out}) {
    int64_t offset = 0;
    SmallVector<int64_t> strides;
    if (failed(t.getStridesAndOffset(strides, offset)) || strides.size() != 2 ||
        strides[1] != 1 || strides[0] != t.getShape()[1])
      return emitOpError("input and output rows must be contiguous and as wide "
                         "as the shape says; the runtime takes no stride");
  }
  // What `sp_tiled_norm` actually implements. It branches on LAYERNORM and
  // SOFTMAX and has no third case, so an iGELU never reaches a mvout and the
  // output buffer is left exactly as it was -- measured on the board, every
  // element zero, where the runtime's own `scale_and_sat` gives real values.
  // iGELU lives on the matmul's own scale pipeline instead, which does
  // configure it; see `matmul_i8_scale`.
  switch (getAct()) {
  case Act::LAYERNORM:
  case Act::SOFTMAX:
    break;
  default:
    return emitOpError("act must be layernorm or softmax; the runtime's norm "
                       "kernel implements no other, and relu belongs to the "
                       "matmul's own scale pipeline");
  }
  return verifySoftmaxScale(*this, getAct(), getScale());
}

/// LayerNorm and Softmax reduce along a row. A convolution's output has no such
/// row -- the accumulator holds a window of pixels, and `tiled_conv_auto` has
/// nowhere to say what to reduce over -- so only the pointwise activations
/// belong on one.
static LogicalResult verifyPointwiseAct(Operation *op, Act act) {
  if (act == Act::NONE || act == Act::RELU)
    return success();
  return op->emitOpError("a convolution can only fuse a pointwise activation; "
                         "layernorm and softmax reduce along a row, which its "
                         "accumulator does not have");
}

LogicalResult Conv2DInt8Op::verify() {
  if (failed(verifyPointwiseAct(*this, getAct())))
    return failure();
  auto filter = llvm::cast<MemRefType>(getFilter().getType());
  if (!filter.hasStaticShape())
    return emitOpError("filter must have a static shape");
  // (KH, KW, C, F); the runtime takes a single kernel_dim.
  if (filter.getShape()[0] != filter.getShape()[1])
    return emitOpError("filter must be square, got ")
           << filter.getShape()[0] << "x" << filter.getShape()[1];

  return verifyConvShapes(
      *this, llvm::cast<MemRefType>(getInput().getType()),
      llvm::cast<MemRefType>(getOutput().getType()), getBias(),
      filter.getShape()[2], filter.getShape()[3], filter.getShape()[0],
      getStride(), getPadding(), getDilation(), getPoolSize(), getPoolStride(),
      getPoolPadding(), getInputDilation());
}

LogicalResult DepthwiseConv2DInt8Op::verify() {
  if (failed(verifyPointwiseAct(*this, getAct())))
    return failure();
  auto filter = llvm::cast<MemRefType>(getFilter().getType());
  if (!filter.hasStaticShape())
    return emitOpError("filter must have a static shape");
  // (C, KH, KW): channel-major, unlike linalg's (KH, KW, C).
  if (filter.getShape()[1] != filter.getShape()[2])
    return emitOpError("filter must be square, got ")
           << filter.getShape()[1] << "x" << filter.getShape()[2];

  int64_t channels = filter.getShape()[0];
  return verifyConvShapes(
      *this, llvm::cast<MemRefType>(getInput().getType()),
      llvm::cast<MemRefType>(getOutput().getType()), getBias(), channels,
      channels, filter.getShape()[1], getStride(), getPadding(), /*dilation=*/1,
      getPoolSize(), getPoolStride(), getPoolPadding());
}

LogicalResult MemsetOp::verify() {
  auto ty = llvm::cast<MemRefType>(getBuffer().getType());
  auto [strides, offset] = ty.getStridesAndOffset();
  int64_t packed = 1;
  for (int d = ty.getRank() - 1; d >= 0; d--) {
    // A dimension of size one never steps, so its stride cannot leave a gap.
    // The slabs `--fill-only-the-border` makes are exactly this shape: the last
    // rows of a buffer, entire, carry the whole buffer's outermost stride over
    // an extent of one.
    if (ty.getDimSize(d) == 1)
      continue;
    if (strides[d] != packed)
      return emitOpError("needs a contiguous buffer; dimension ")
             << d << " has stride " << strides[d] << " where " << packed
             << " was expected";
    packed *= ty.getDimSize(d);
  }
  if (!ty.getElementType().isIntOrFloat() ||
      ty.getElementType().getIntOrFloatBitWidth() % 8 != 0)
    return emitOpError("needs an element type that is a whole number of bytes");
  return success();
}
