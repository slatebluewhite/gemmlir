//===- PointwiseConvToMatmulPass.cpp ---------------------------*- C++ -*-===//
//
// A 1x1 convolution with nowhere to requantize is a matmul.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_POINTWISECONVTOMATMUL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The accelerator's convolution always requantizes to i8 -- `tiled_conv_auto`
/// writes `elem_t` and there is no other form -- so a convolution whose result
/// the model returns has nothing to fold into and stays a scalar loop. A 1x1
/// convolution over NHWC is a matmul with the pixels as rows, and the matmul
/// call *does* write the i32 accumulator, so that one offloads.
class PointwiseToMatmul : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp conv,
                                PatternRewriter &rewriter) const final {
    if (conv.getInputs().size() != 2 || conv.getOutputs().size() != 1 ||
        conv->getNumResults() != 1)
      return failure();
    auto inTy = dyn_cast<RankedTensorType>(conv.getInputs()[0].getType());
    auto filterTy = dyn_cast<RankedTensorType>(conv.getInputs()[1].getType());
    auto outTy = dyn_cast<RankedTensorType>(conv->getResult(0).getType());
    if (!inTy || !filterTy || !outTy || !inTy.hasStaticShape() ||
        !filterTy.hasStaticShape() || !outTy.hasStaticShape())
      return failure();
    if (inTy.getRank() != 4 || filterTy.getRank() != 4 || outTy.getRank() != 4)
      return failure();
    if (!inTy.getElementType().isInteger(8) ||
        !filterTy.getElementType().isInteger(8) ||
        !outTy.getElementType().isInteger(32))
      return failure();

    // One tap, one step: then every output pixel reads exactly its own input
    // pixel and the convolution is a contraction over the channels.
    if (filterTy.getDimSize(0) != 1 || filterTy.getDimSize(1) != 1)
      return failure();
    auto unit = [](DenseIntElementsAttr a) {
      return a && a.getNumElements() == 2 &&
             llvm::all_of(a.getValues<APInt>(), [](APInt v) { return v.isOne(); });
    };
    if (!unit(conv.getStrides()) || !unit(conv.getDilations()))
      return failure();
    if (inTy.getDimSize(0) != outTy.getDimSize(0) ||
        inTy.getDimSize(1) != outTy.getDimSize(1) ||
        inTy.getDimSize(2) != outTy.getDimSize(2))
      return failure();

    // Only where it would not fold as a convolution. A requantization after it
    // becomes `conv2d_i8`, which takes the bias and the activation with it and
    // is the faster call on this board.
    if (!conv->hasOneUse())
      return failure();
    Operation *user = *conv->getUsers().begin();
    if (user->getNumResults() != 1 ||
        !getElementTypeOrSelf(user->getResult(0).getType()).isF32())
      return failure();

    Location loc = conv.getLoc();
    int64_t pixels = inTy.getDimSize(0) * inTy.getDimSize(1) * inTy.getDimSize(2);
    int64_t channels = inTy.getDimSize(3), filters = outTy.getDimSize(3);
    if (filterTy.getDimSize(2) != channels || filterTy.getDimSize(3) != filters)
      return failure();

    SmallVector<ReassociationIndices> pixelRows = {{0, 1, 2}, {3}};
    SmallVector<ReassociationIndices> tap = {{0, 1, 2}, {3}};
    Value rows = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get({pixels, channels}, inTy.getElementType()),
        conv.getInputs()[0], pixelRows);
    Value weights = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get({channels, filters}, filterTy.getElementType()),
        conv.getInputs()[1], tap);
    Value init = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get({pixels, filters}, outTy.getElementType()),
        conv.getOutputs()[0], pixelRows);

    auto matmulTy = RankedTensorType::get({pixels, filters}, outTy.getElementType());
    Value product = rewriter
                        .create<linalg::MatmulOp>(loc, TypeRange{matmulTy},
                                                  ValueRange{rows, weights},
                                                  ValueRange{init})
                        .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(conv, outTy, product,
                                                       pixelRows);
    return success();
  }
};

class PointwiseConvToMatmul
    : public impl::PointwiseConvToMatmulBase<PointwiseConvToMatmul> {
public:
  using impl::PointwiseConvToMatmulBase<PointwiseConvToMatmul>::PointwiseConvToMatmulBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PointwiseToMatmul>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
