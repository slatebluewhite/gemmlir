//===- MaterializePadSourcesPass.cpp ---------------------------*- C++ -*-===//
//
// Keep a padded convolution's input out of the padding's own buffer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

#include <functional>

namespace mlir::gemmlir {

#define GEN_PASS_DEF_MATERIALIZEPADSOURCES
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// Bufferization will write a `tensor.pad`'s source straight into the middle of
/// the padded buffer when it can. The runtime cannot address that window, so
/// say that the source wants a buffer of its own.
class MaterializeSource : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp pad,
                                PatternRewriter &rewriter) const final {
    auto srcTy = dyn_cast<RankedTensorType>(pad.getSource().getType());
    auto padTy = dyn_cast<RankedTensorType>(pad.getType());
    if (!srcTy || !padTy || !srcTy.hasStaticShape() || !padTy.hasStaticShape())
      return failure();
    // Only what the accelerator's convolution reads: NHWC integers, padded on
    // the two spatial axes alone.
    if (srcTy.getRank() != 4 || !srcTy.getElementType().isInteger(8))
      return failure();
    if (pad.getSourceType().getShape()[0] != padTy.getShape()[0] ||
        pad.getSourceType().getShape()[3] != padTy.getShape()[3])
      return failure();

    // Already in a buffer of its own, or not something bufferization would put
    // in the padding's.
    if (pad.getSource().getDefiningOp<bufferization::AllocTensorOp>())
      return failure();
    auto producer = pad.getSource().getDefiningOp<linalg::LinalgOp>();
    if (!producer)
      return failure();
    // Only where bufferization would take the chance. With another consumer it
    // has to keep the producer's own buffer anyway, and asking for one here
    // just adds a copy -- nine of them on ResNet-20, whose every block hands
    // its result to the shortcut as well.
    if (!pad.getSource().hasOneUse())
      return failure();

    // And only where it stands in the way: a convolution that could take this
    // padding itself. A grouped convolution reaches its padding through one
    // `tensor.extract_slice` per group, so look through those -- without it the
    // whole grouped family kept its pointwise convolution in software, because
    // its requantization was writing into the padded buffer's middle and the
    // accelerator cannot address that window (`out_stride` is one stride per
    // pixel; the rows of a window narrower than its buffer do not follow it).
    std::function<bool(Operation *)> reachesConv = [&](Operation *user) {
      if (isa<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(user))
        return true;
      if (!isa<tensor::ExtractSliceOp>(user))
        return false;
      return llvm::any_of(user->getUsers(), reachesConv);
    };
    // The other way it stands in the way: whatever *produced* the source could
    // have folded into the accelerator, and cannot while its result is written
    // into the padding's middle. `gup`'s pointwise convolution is that -- its
    // groups have already become im2col packs by this point, so there is no
    // convolution below the padding to find, and the one above it is the one
    // being kept in software.
    auto foldableAbove = [&] {
      for (Value in : producer.getDpsInputs()) {
        Operation *def = in.getDefiningOp();
        if (!def)
          continue;
        if (isa<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(def))
          return true;
        if (auto contraction = dyn_cast<linalg::LinalgOp>(def))
          if (linalg::isaContractionOpInterface(contraction))
            return true;
      }
      return false;
    };
    if (!llvm::any_of(pad->getUsers(), reachesConv) && !foldableAbove())
      return failure();

    Value owned = rewriter.create<bufferization::AllocTensorOp>(
        pad.getLoc(), srcTy, ValueRange{}, pad.getSource());
    rewriter.modifyOpInPlace(pad,
                             [&] { pad.getSourceMutable().assign(owned); });
    return success();
  }
};

class MaterializePadSources
    : public impl::MaterializePadSourcesBase<MaterializePadSources> {
public:
  using impl::MaterializePadSourcesBase<
      MaterializePadSources>::MaterializePadSourcesBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<bufferization::BufferizationDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MaterializeSource>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
