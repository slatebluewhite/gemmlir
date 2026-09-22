//===- ConvToImg2ColPass.cpp -------------------------------*- C++ -*-===//
//
// Rewrites convolutions as im2col packing plus a matmul.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_CONVTOIMG2COL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// im2col for NHWC, with the patch offset left as nested loops.
///
/// MLIR's own rewrite packs into `N x P x K`, where `K` is the patch offset
/// `kh*KW*C + kw*C + c` and `P` is the position `oh*OW + ow`. Taking those apart
/// again costs a floordiv and a mod per level, and which loop they land on
/// decides how often: with `K` innermost -- which is where NHWC puts it -- the
/// divisors are the kernel's, so four multiply-shift sequences run on every one
/// of the 6912 elements of the CNN's first pack. Measured on the board, that
/// pack takes 9.21 ms; iterating it position-innermost instead takes 4.13 ms,
/// and the NCHW pack, whose divisors are powers of two, takes 3.79 ms.
///
/// Writing the loops out -- `(n, oh, ow, kh, kw, c)`, with the indices *being*
/// the offsets -- removes the arithmetic rather than moving it: **1.77 ms**, a
/// little over twice as fast as the NCHW pack this replaces. A `collapse_shape`
/// afterwards is a view, so everything downstream sees exactly what MLIR's
/// rewrite produced.
///
/// Only a batch of one is handled here; anything else falls through to MLIR's
/// pattern, which is correct and general.
/// The one shape the accelerator's convolution is known to compute wrong.
/// See the note in `matchAndRewrite` for how it was measured.
/// True when everything in this operation's body is something the accelerator's
/// own output pipeline can do.
///
/// A `conv2d_i8` writes **i8**: its tail has to be
/// `saturate(scale * accumulator + bias)` with at most a relu, which is what
/// `matchRequantize` takes. Anything else in the way -- a transcendental above
/// all -- means the convolution has no requantization to fold into however much
/// the value below it looks like one, and it belongs on the matmul path
/// instead, where `matmul_i8` leaves an i32 accumulator for the tail to read.
///
/// EfficientNet is the case: **SiLU**, `x * sigmoid(x)`, sits between every
/// convolution and its quantization, and `math.exp` is not something the mvout
/// pipeline has. A whitelist rather than a blacklist -- being wrong here costs
/// a packed convolution that need not have been, never a wrong answer.
static bool bodyIsAbsorbable(Operation *op) {
  auto generic = llvm::dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return true; // not an elementwise tail at all; the walk judges it elsewhere
  // Tried and measured: calling a **reduction** unabsorbable here -- which it
  // is, the accelerator writes its own output and a layer norm's mean does not
  // have that shape -- moves two of EfficientNet's convolutions off `conv2d_i8`
  // and onto the matmul path, **1114 -> 1373 ms**, and buys ConvNeXt nothing
  // (its depthwise convolutions are rescued by --quantize-unfoldable-tails, not
  // by packing). The walk judges a reduction elsewhere.
  for (Operation &inner : generic.getRegion().front())
    if (!llvm::isa<arith::SIToFPOp, arith::FPToSIOp, arith::TruncIOp,
                   arith::ExtSIOp, arith::TruncFOp, arith::ExtFOp,
                   arith::MulFOp, arith::DivFOp, arith::AddFOp, arith::SubFOp,
                   arith::NegFOp, arith::AddIOp, arith::SubIOp, arith::MaxSIOp,
                   arith::MinSIOp, arith::MaximumFOp, arith::MinimumFOp,
                   arith::MaxNumFOp, arith::MinNumFOp, arith::CmpFOp,
                   arith::CmpIOp, arith::SelectOp, arith::ConstantOp,
                   math::RoundEvenOp, linalg::YieldOp>(&inner))
      return false;
  return true;
}

static bool acceleratorIsWrong(linalg::Conv2DNhwcHwcfOp conv) {
  if (conv.getInputs().size() != 2 || conv->getNumResults() != 1)
    return false;
  auto inTy = llvm::dyn_cast<RankedTensorType>(conv.getInputs()[0].getType());
  auto oTy = llvm::dyn_cast<RankedTensorType>(conv->getResult(0).getType());
  if (!inTy || !oTy || inTy.getRank() != 4 || oTy.getRank() != 4 ||
      !inTy.hasStaticShape() || !oTy.hasStaticShape())
    return false;
  auto fTy = llvm::dyn_cast<RankedTensorType>(conv.getInputs()[1].getType());
  if (!fTy || fTy.getRank() != 4 || fTy.getDimSize(0) != 3 ||
      fTy.getDimSize(1) != 3)
    return false;
  auto unit = [](DenseIntElementsAttr a) {
    return a && a.getNumElements() == 2 &&
           llvm::all_of(a.getValues<APInt>(), [](APInt v) { return v.isOne(); });
  };
  if (!unit(conv.getStrides()) || !unit(conv.getDilations()))
    return false;
  return oTy.getDimSize(1) == 24 && oTy.getDimSize(2) == 24 &&
         inTy.getDimSize(3) <= 5;
}

/// A convolution the accelerator's own op cannot say.
///
/// `tiled_conv_auto` takes **one** integer for the kernel, one for the stride
/// and one for the dilation, so `Conv2DInt8Op`'s verifier refuses anything
/// whose two spatial extents differ -- a 1x3, a 3x1, a stride of (2, 1). Such a
/// convolution has no call to fold into and would otherwise stay a scalar loop
/// over every multiply: a separable 3x3 written as a 1x3 and a 3x1, on a
/// 16x16 image with eight channels, is 67,600 of them.
///
/// A matmul has no such restriction. Packing it is the same trade the grouped
/// family makes -- more elements moved, all of the arithmetic on the
/// accelerator.
static bool acceleratorCannotExpress(linalg::Conv2DNhwcHwcfOp conv) {
  if (conv.getInputs().size() != 2)
    return false;
  auto fTy = llvm::dyn_cast<RankedTensorType>(conv.getInputs()[1].getType());
  if (!fTy || fTy.getRank() != 4 || !fTy.hasStaticShape())
    return false;
  if (fTy.getDimSize(0) != fTy.getDimSize(1))
    return true;
  auto bothTheSame = [](DenseIntElementsAttr a) {
    if (!a || a.getNumElements() != 2)
      return true;
    auto values = llvm::to_vector(a.getValues<APInt>());
    return values[0] == values[1];
  };
  return !bothTheSame(conv.getStrides()) || !bothTheSame(conv.getDilations());
}

class NhwcConvToSplitImg2Col
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  NhwcConvToSplitImg2Col(MLIRContext *ctx, bool unfoldableOnly)
      : OpRewritePattern(ctx, /*benefit=*/2), unfoldableOnly(unfoldableOnly) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp conv,
                                PatternRewriter &rewriter) const final {
    // `tiled_conv_auto` writes `elem_t` and there is no other form, so a
    // convolution whose result is not requantized has nothing to fold into and
    // stays a scalar loop -- a grouped convolution at the end of a block is
    // exactly that, and the whole grouped family had no accelerator call for
    // its convolutions at all. The matmul call *does* write the i32
    // accumulator, so packing first is what makes those offload. Everything
    // that would fold is left alone: `conv2d_i8` takes the bias, the activation
    // and the pooling with it and is the faster call here.
    if (unfoldableOnly) {
      if (!llvm::isa<RankedTensorType>(conv->getResult(0).getType()))
        return failure();
      if (!getElementTypeOrSelf(conv.getInputs()[0].getType()).isInteger(8) ||
          !getElementTypeOrSelf(conv->getResult(0).getType()).isInteger(32))
        return failure();
      // "Nothing to fold into" is not the same as "the next operation is in
      // f32": a residual add and a global pool sit between a ResNet block's
      // last convolution and the requantization of the block's output, and
      // `--convert-linalg-to-gemmlir` reads that whole shape. Follow the
      // result down instead, and pack only where it reaches the function's
      // return still wide. Anything else -- a branch, a chain longer than this
      // walk, an i8 on the way -- is left to the convolution path.
      Value wide = conv->getResult(0);
      // The accelerator writes exactly its own output. If something on the way
      // down makes *more* elements than the convolution produced, whatever
      // requantization follows belongs to that, not to this convolution --
      // so there is nothing here to fold into after all. A decoder's
      // nearest-neighbour upsample is the case: `conv -> dequantize ->
      // upsample -> quantize` ends in an i8, and the walk below would take
      // that as proof the convolution folds, while the requantize is four
      // times the size of anything it wrote. A residual add keeps the count
      // and a global pool lowers it; neither trips this.
      auto elementsOf = [](Value v) -> int64_t {
        auto t = llvm::dyn_cast<RankedTensorType>(v.getType());
        return t && t.hasStaticShape() ? t.getNumElements() : -1;
      };
      int64_t produced = elementsOf(wide);
      bool grew = false;
      bool returned = false;
      for (unsigned step = 0; step < 8 && !returned; step++) {
        // Tried and measured: a layer norm reads its input twice, so stopping
        // at the branch leaves ConvNeXt's patchify convolution a scalar loop --
        // but packing whenever *any* user below a branch is unabsorbable took
        // two of EfficientNet's convolutions off `conv2d_i8` and onto the
        // matmul path, and that model went **1114 -> 1373 ms**. A branch is not
        // by itself evidence that the convolution's own tail is gone. One
        // scalar convolution in ConvNeXt is the cheaper side of that trade.
        if (!wide.hasOneUse())
          break;
        Operation *user = *wide.getUsers().begin();
        if (user->hasTrait<OpTrait::ReturnLike>()) {
          returned = true;
          break;
        }
        // A pool while the values are still wide is as final as the return,
        // and for the same reason: `conv2d_i8` writes its own output buffer, so
        // it cannot produce a pooled one. `FoldMaxPoolIntoConv` does attach a
        // pool to a call, but only to a call that already exists -- which needs
        // the tail to have become a requantization first, and a tail that is
        // still f32 *here* is one that never will be.
        //
        // DenseNet-121's stem is that shape: a 7x7 stride-2 convolution on three
        // channels whose f32 tail feeds a max-pool, and whose pooled result has
        // seven users, so the branch test below stopped the walk one step too
        // late and left the whole convolution a scalar loop. Sampling the
        // program counter put **72% of the entire model** in it.
        //
        // Three input channels is also next door to the corner the accelerator
        // gets wrong ([[gemmini-conv-wrong-at-24x24]]), and im2col plus a matmul
        // is the route that agrees with the host everywhere it has been checked.
        if (llvm::isa<linalg::PoolingNhwcMaxOp, linalg::PoolingNchwMaxOp,
                      linalg::PoolingNhwcSumOp>(user)) {
          returned = true;
          break;
        }
        // A join while the values are still wide is as final as the return.
        // `conv2d_i8` writes one window of one buffer; it cannot write a
        // branch's slice of a concatenation in f32, so the tail below this
        // convolution is never going to become a requantization it can take.
        // ShuffleNet's unit is exactly that -- its two branches are joined in
        // f32 and quantized only after the channel shuffle -- and the 3x3
        // branch was the last scalar convolution in the model set.
        if (llvm::isa<tensor::ConcatOp, tensor::InsertSliceOp>(user)) {
          returned = true;
          break;
        }
        if (user->getNumResults() != 1)
          break;
        // A tail the mvout pipeline cannot do is as final as a return: there is
        // nothing here for the convolution to fold into.
        if (!bodyIsAbsorbable(user)) {
          returned = true;
          break;
        }
        if (getElementTypeOrSelf(user->getResult(0).getType()).isInteger(8)) {
          // An i8 below a growth is somebody else's requantization.
          returned = grew;
          break;
        }
        wide = user->getResult(0);
        int64_t here = elementsOf(wide);
        if (produced < 0 || here < 0)
          break;
        grew |= here > produced;
      }
      // ... unless the accelerator gets this one wrong. Measured on the U280
      // board with `gemmlir_rt.o`, an exact integer reference and one
      // convolution per process: a 3x3 stride-1 convolution with a 24x24 result
      // and at most five input channels writes **one** output pixel wrong --
      // (22, 23), every channel of it, plausible values rather than zeros or a
      // stale line. `conv_cpu` is exact on all 488 shapes swept (4..64 square,
      // 1..8 in-channels), and so is every other size from 4 to 64 on the
      // accelerator; it is this shape alone. The tiling chosen for a 24x24
      // result is 22 rows by 23 columns, which leaves a corner tile two rows by
      // one column, and the wrong pixel is the first of it.
      //
      // It kept `atr` and `atrn` quietly wrong for weeks: both sit at 0.0152
      // and 0.0150 relative L2 whether the convolutions run on the accelerator
      // or on the host, so only a byte-for-byte comparison against the CPU
      // runtime shows it (`scripts/board-sweep.sh`).
      //
      // im2col plus a matmul computes the same thing and agrees with the host
      // everywhere it has been checked, so a convolution in that corner goes
      // that way instead.
      if (!returned && !acceleratorIsWrong(conv) &&
          !acceleratorCannotExpress(conv))
        return failure();
    }
    if (conv.getInputs().size() != 2 || conv.getOutputs().size() != 1 ||
        conv->getNumResults() != 1)
      return failure();
    auto inTy = llvm::dyn_cast<RankedTensorType>(conv.getInputs()[0].getType());
    auto fTy = llvm::dyn_cast<RankedTensorType>(conv.getInputs()[1].getType());
    auto oTy = llvm::dyn_cast<RankedTensorType>(conv.getOutputs()[0].getType());
    if (!inTy || !fTy || !oTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !oTy.hasStaticShape())
      return failure();
    if (inTy.getRank() != 4 || fTy.getRank() != 4 || oTy.getRank() != 4)
      return failure();
    if (oTy.getShape()[0] != 1)
      return failure();

    int64_t oh = oTy.getShape()[1], ow = oTy.getShape()[2];
    int64_t kh = fTy.getShape()[0], kw = fTy.getShape()[1];
    int64_t c = fTy.getShape()[2], f = fTy.getShape()[3];
    if (inTy.getShape()[3] != c)
      return failure();

    auto twoOf = [](DenseIntElementsAttr a, unsigned i) {
      return (*(a.value_begin<APInt>() + i)).getSExtValue();
    };
    int64_t sh = twoOf(conv.getStrides(), 0), sw = twoOf(conv.getStrides(), 1);
    int64_t dh = twoOf(conv.getDilations(), 0), dw = twoOf(conv.getDilations(), 1);

    Location loc = conv.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr n, y, x, i, j, ch;
    bindDims(ctx, n, y, x, i, j, ch);
    AffineMap read = AffineMap::get(
        6, 0, {n, y * sh + i * dh, x * sw + j * dw, ch}, ctx);
    AffineMap write = AffineMap::getMultiDimIdentityMap(6, ctx);

    auto colTy = RankedTensorType::get({int64_t(1), oh, ow, kh, kw, c},
                                       inTy.getElementType());
    Value colInit = rewriter.create<tensor::EmptyOp>(loc, colTy.getShape(),
                                                     colTy.getElementType());
    SmallVector<utils::IteratorType> iters(6, utils::IteratorType::parallel);
    auto pack = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{colTy}, ValueRange{conv.getInputs()[0]},
        ValueRange{colInit}, ArrayRef<AffineMap>{read, write}, iters,
        [](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, args[0]);
        });

    SmallVector<ReassociationIndices> colGroups = {{0, 1, 2}, {3, 4, 5}};
    SmallVector<ReassociationIndices> lastAlone = {{0, 1, 2}, {3}};
    Value cols = rewriter.create<tensor::CollapseShapeOp>(
        loc, pack.getResult(0), colGroups);
    Value weights = rewriter.create<tensor::CollapseShapeOp>(
        loc, conv.getInputs()[1], lastAlone);
    Value init = rewriter.create<tensor::CollapseShapeOp>(
        loc, conv.getOutputs()[0], lastAlone);

    auto accTy = RankedTensorType::get({oh * ow, f}, oTy.getElementType());
    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, TypeRange{accTy}, ValueRange{cols, weights}, ValueRange{init});
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        conv, oTy, matmul.getResult(0), lastAlone);
    return success();
  }

private:
  bool unfoldableOnly;
};

class ConvToImg2Col : public impl::ConvToImg2ColBase<ConvToImg2Col> {
public:
  using impl::ConvToImg2ColBase<ConvToImg2Col>::ConvToImg2ColBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<NhwcConvToSplitImg2Col>(&getContext(), unfoldableOnly);
    // MLIR's own patterns take any convolution they are given; with the guard
    // on they would undo the point of it.
    if (!unfoldableOnly)
      linalg::populateConvertConv2DToImg2ColPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
