//===- AveragePoolToContractionPass.cpp ----------------------------*- C++ -*-===//
//
// A whole-image sum pool is a contraction against a constant.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_AVERAGEPOOLTOCONTRACTION
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// `adaptive_avg_pool2d(x, 1)` arrives as a sum over the whole image followed
/// by a divide by the pixel count. Summing a whole image per channel is a
/// contraction: with the image read as `(H*W, C)` it is `ones(1, H*W) x image`,
/// and that form goes down the path that already exists -- the constant is
/// folded to i8 once, the sum accumulates in i32, and the divide by the count
/// is absorbed into the requantization the accelerator call does anyway.
///
/// The alternative, Gemmini's own `tiled_global_average_auto`, produces the
/// mean at the *input's* scale. For a mean over hundreds of pixels the range
/// collapses by roughly that factor, so most of the output's int8 range would
/// go unused; this way the result is requantized at the scale the calibration
/// measured for it.
class GlobalSumPoolToMatmul : public OpRewritePattern<linalg::PoolingNhwcSumOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNhwcSumOp pool,
                                PatternRewriter &rewriter) const final {
    if (pool.getInputs().size() != 2 || pool.getOutputs().size() != 1)
      return failure();
    auto inTy = dyn_cast<RankedTensorType>(pool.getInputs()[0].getType());
    auto outTy = dyn_cast<RankedTensorType>(pool.getOutputs()[0].getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return failure();
    if (inTy.getRank() != 4 || outTy.getRank() != 4)
      return failure();
    if (!inTy.getElementType().isF32())
      return failure();

    // The window is the whole image: the result has a single pixel per image and
    // the channels are untouched.
    if (inTy.getDimSize(0) != outTy.getDimSize(0))
      return failure();
    if (outTy.getDimSize(1) != 1 || outTy.getDimSize(2) != 1)
      return failure();
    if (outTy.getDimSize(3) != inTy.getDimSize(3))
      return failure();
    // The window is the image. Nothing needs to be said about dilation or
    // stride: a window as wide as the image can only be dilated by one without
    // reaching past the edge, and one output pixel leaves the stride nothing to
    // step over.
    auto windowTy = dyn_cast<RankedTensorType>(pool.getInputs()[1].getType());
    if (!windowTy || windowTy.getRank() != 2 ||
        windowTy.getDimSize(0) != inTy.getDimSize(1) ||
        windowTy.getDimSize(1) != inTy.getDimSize(2))
      return failure();

    // The sum starts from zero, or there is a value here the matmul would drop.
    auto fill = pool.getOutputs()[0].getDefiningOp<linalg::FillOp>();
    if (!fill || fill.getInputs().size() != 1 ||
        !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
      return failure();

    Location loc = pool.getLoc();
    Type elem = inTy.getElementType();
    int64_t images = inTy.getDimSize(0);
    int64_t pixels = inTy.getDimSize(1) * inTy.getDimSize(2);
    int64_t channels = inTy.getDimSize(3);

    // The image as a matrix: NHWC already has the channel innermost, so the
    // pixels are the rows and nothing moves. One image collapses to a matrix
    // and contracts; a batch of them keeps the batch dimension, because summing
    // across the collapse would sum across images.
    bool batched = images != 1;
    SmallVector<ReassociationIndices> asRows =
        batched ? SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}}
                : SmallVector<ReassociationIndices>{{0, 1, 2}, {3}};
    SmallVector<int64_t> imageShape =
        batched ? SmallVector<int64_t>{images, pixels, channels}
                : SmallVector<int64_t>{pixels, channels};
    Value image = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get(imageShape, elem), pool.getInputs()[0],
        asRows);

    SmallVector<int64_t> onesShape =
        batched ? SmallVector<int64_t>{images, 1, pixels}
                : SmallVector<int64_t>{1, pixels};
    auto onesTy = RankedTensorType::get(onesShape, elem);
    Value ones = rewriter.create<arith::ConstantOp>(
        loc, onesTy,
        DenseElementsAttr::get(onesTy, rewriter.getFloatAttr(elem, 1.0)));

    SmallVector<int64_t> sumShape = batched
                                        ? SmallVector<int64_t>{images, 1, channels}
                                        : SmallVector<int64_t>{1, channels};
    auto sumTy = RankedTensorType::get(sumShape, elem);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elem, 0.0));
    Value init = rewriter.create<tensor::EmptyOp>(loc, sumTy.getShape(), elem);
    Value zeroed =
        rewriter.create<linalg::FillOp>(loc, zero, init).getResult(0);
    Value sum =
        batched ? rewriter
                      .create<linalg::BatchMatmulOp>(loc, TypeRange{sumTy},
                                                     ValueRange{ones, image},
                                                     ValueRange{zeroed})
                      .getResult(0)
                : rewriter
                      .create<linalg::MatmulOp>(loc, TypeRange{sumTy},
                                                ValueRange{ones, image},
                                                ValueRange{zeroed})
                      .getResult(0);

    // Back into the shape the divide that follows is written over.
    SmallVector<ReassociationIndices> asImage =
        batched ? SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}}
                : SmallVector<ReassociationIndices>{{0, 1, 2}, {3}};
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(pool, outTy, sum,
                                                       asImage);
    return success();
  }
};

/// The calibrated range of whatever produced `v`.
///
/// An average never leaves the range of what it averages, so the contraction
/// this pass makes out of a windowed pool can be quantized at its input's
/// scale. Without it the operation is unannotated and falls back to the pass
/// option's fixed scale, which is the same fixed-scale problem calibration
/// exists to solve: on `apb` it was 0.0045 relative L2 against 0.0140.
///
/// Only for a window. A global mean over hundreds of pixels collapses the range
/// by roughly that factor, so the input's scale would leave most of the output's
/// int8 range unused, and the calibration measured for the result is the right
/// answer there.
static Attribute inputActivationScale(Value v) {
  // The producer is rarely the annotated operation itself: a relu, a bias or a
  // batch-norm remnant usually sits between. None of those widens the range --
  // the calibration measured the layer's own output, before them -- so the
  // first annotation above is an upper bound, which is what a scale has to be.
  for (unsigned step = 0; step < 8; step++) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return {};
    if (Attribute a = def->getAttr("gemmlir.activation_scale"))
      return a;
    auto generic = llvm::dyn_cast<linalg::GenericOp>(def);
    if (!generic || generic.getInputs().empty())
      return {};
    v = generic.getInputs()[0];
  }
  return {};
}

/// An average pool with a window is a **depthwise convolution** whose filter is
/// all ones: `out[n, oh, ow, c] = sum over the window of in[...][c]`, which is
/// what `linalg.depthwise_conv_2d_nhwc_hwc` computes with a filter of ones, and
/// what `tiled_conv_dw_auto` runs.
///
/// It matters for the same reason the global one does. A frontend sums in f32
/// and divides afterwards, so the convolution feeding the pool has no i8 result
/// to end in and stays a scalar loop -- and unlike a max pool, a sum does not
/// commute with a quantization, so `--requantize-before-pooling` cannot move
/// the requantization across it. As a depthwise convolution it is quantized
/// like any other layer, the ones fold to i8 at compile time, and the divide by
/// the window's size disappears into the call's requantization scale.
class WindowedSumPoolToDepthwise
    : public OpRewritePattern<linalg::PoolingNhwcSumOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNhwcSumOp pool,
                                PatternRewriter &rewriter) const final {
    if (pool.getInputs().size() != 2 || pool.getOutputs().size() != 1)
      return failure();
    auto inTy = dyn_cast<RankedTensorType>(pool.getInputs()[0].getType());
    auto outTy = dyn_cast<RankedTensorType>(pool.getOutputs()[0].getType());
    auto windowTy = dyn_cast<RankedTensorType>(pool.getInputs()[1].getType());
    if (!inTy || !outTy || !windowTy || !inTy.hasStaticShape() ||
        !outTy.hasStaticShape() || !windowTy.hasStaticShape())
      return failure();
    if (inTy.getRank() != 4 || outTy.getRank() != 4 || windowTy.getRank() != 2)
      return failure();
    if (!inTy.getElementType().isF32())
      return failure();
    // The whole image is the other pattern's: a matmul keeps the result at the
    // scale the calibration measured for it, where a filter this size would not
    // fit the accelerator's window.
    if (outTy.getDimSize(1) == 1 && outTy.getDimSize(2) == 1)
      return failure();

    // The sum starts from zero, or there is a value here the convolution would
    // drop.
    auto fill = pool.getOutputs()[0].getDefiningOp<linalg::FillOp>();
    if (!fill || fill.getInputs().size() != 1 ||
        !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
      return failure();

    Location loc = pool.getLoc();
    Type elem = inTy.getElementType();
    int64_t channels = inTy.getDimSize(3);
    auto filterTy = RankedTensorType::get(
        {windowTy.getDimSize(0), windowTy.getDimSize(1), channels}, elem);
    Value ones = rewriter.create<arith::ConstantOp>(
        loc, filterTy,
        DenseElementsAttr::get(filterTy, rewriter.getFloatAttr(elem, 1.0)));

    auto conv = rewriter.create<linalg::DepthwiseConv2DNhwcHwcOp>(
        loc, TypeRange{outTy}, ValueRange{pool.getInputs()[0], ones},
        ValueRange{pool.getOutputs()[0]}, pool.getStrides(),
        pool.getDilations());
    if (Attribute scale = inputActivationScale(pool.getInputs()[0]))
      conv->setAttr("gemmlir.activation_scale", scale);
    rewriter.replaceOp(pool, conv.getResults());
    return success();
  }
};

class AveragePoolToContraction
    : public impl::AveragePoolToContractionBase<AveragePoolToContraction> {
public:
  using impl::AveragePoolToContractionBase<AveragePoolToContraction>::AveragePoolToContractionBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<GlobalSumPoolToMatmul, WindowedSumPoolToDepthwise>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
