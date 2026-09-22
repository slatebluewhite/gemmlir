//===- DropUnreadPaddingPass.cpp --------------------------------*- C++ -*-===//
//
// Drop a padding a pooling operation never reads.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_DROPUNREADPADDING
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The spatial axes of a pooling operation, in the order the shapes carry them.
static bool spatialAxes(Operation *op, SmallVectorImpl<unsigned> &axes) {
  if (llvm::isa<linalg::PoolingNhwcMaxOp, linalg::PoolingNhwcMaxUnsignedOp,
                linalg::PoolingNhwcMinOp, linalg::PoolingNhwcSumOp>(op)) {
    axes.assign({1, 2});
    return true;
  }
  if (llvm::isa<linalg::PoolingNchwMaxOp, linalg::PoolingNchwSumOp>(op)) {
    axes.assign({2, 3});
    return true;
  }
  return false;
}

static SmallVector<int64_t> ints(DenseIntElementsAttr attr) {
  SmallVector<int64_t> out;
  for (const llvm::APInt &v : attr)
    out.push_back(v.getSExtValue());
  return out;
}

/// PyTorch's `ceil_mode` asks for an output one step wider than the input
/// supports, and torch-mlir pays for it with a `tensor.pad` of `stride - 1` on
/// the high side. When the window happens to divide evenly the ceiling and the
/// floor agree, the extra row is never reached, and what is left is a buffer
/// the host zero-fills and copies into for nothing -- and, worse, a copy
/// standing between the convolution and the pool, which is what stops the pool
/// riding out on the convolution's own `mvout`.
///
/// SqueezeNet has three: 31x31 to 33x33, 15x15 to 17x17, 9x9 to 11x11, and not
/// one padded element is read.
///
/// The test is arithmetic and exact. The highest index the pool reads on a
/// spatial axis is `(out - 1) * stride + (window - 1) * dilation`; if that is
/// inside the source on every axis and nothing is padded on the low side, the
/// pad contributes nothing to the result and the pool can read the source.
class DropPaddingNoPoolReads : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp pool,
                                PatternRewriter &rewriter) const final {
    SmallVector<unsigned> axes;
    if (!spatialAxes(pool, axes) || pool.getDpsInputs().size() != 2)
      return failure();
    auto pad = pool.getDpsInputs()[0].getDefiningOp<tensor::PadOp>();
    if (!pad || pad.getNofold())
      return failure();

    auto srcTy = llvm::dyn_cast<RankedTensorType>(pad.getSource().getType());
    auto padTy = llvm::dyn_cast<RankedTensorType>(pad.getType());
    auto winTy = llvm::dyn_cast<RankedTensorType>(pool.getDpsInputs()[1].getType());
    auto outTy = llvm::dyn_cast<RankedTensorType>(pool->getResult(0).getType());
    if (!srcTy || !padTy || !winTy || !outTy || !srcTy.hasStaticShape() ||
        !padTy.hasStaticShape() || !winTy.hasStaticShape() ||
        !outTy.hasStaticShape())
      return failure();

    // A low pad shifts every index, so the reasoning below would have to shift
    // with it; a convolution's border arrives that way and is folded elsewhere.
    for (OpFoldResult low : pad.getMixedLowPad()) {
      auto v = getConstantIntValue(low);
      if (!v || *v != 0)
        return failure();
    }
    // Every axis the pool does not walk has to be unpadded outright: this
    // rewrite only argues about the two it does.
    SmallVector<bool> spatial(padTy.getRank(), false);
    for (unsigned a : axes) {
      if (a >= (unsigned)padTy.getRank())
        return failure();
      spatial[a] = true;
    }
    for (int64_t d = 0; d < padTy.getRank(); d++)
      if (!spatial[d] && padTy.getDimSize(d) != srcTy.getDimSize(d))
        return failure();

    auto strides = pool->getAttrOfType<DenseIntElementsAttr>("strides");
    auto dilations = pool->getAttrOfType<DenseIntElementsAttr>("dilations");
    if (!strides || !dilations)
      return failure();
    SmallVector<int64_t> s = ints(strides), d = ints(dilations);
    if (s.size() != axes.size() || d.size() != axes.size())
      return failure();

    bool anyPadded = false;
    for (auto [i, axis] : llvm::enumerate(axes)) {
      int64_t src = srcTy.getDimSize(axis);
      if (padTy.getDimSize(axis) < src)
        return failure();
      if (padTy.getDimSize(axis) > src)
        anyPadded = true;
      // The window's extents follow the pool's own axis order.
      int64_t window = winTy.getDimSize(i);
      int64_t out = outTy.getDimSize(axis);
      if (out < 1 || window < 1)
        return failure();
      if ((out - 1) * s[i] + (window - 1) * d[i] >= src)
        return failure();
    }
    if (!anyPadded)
      return failure();

    rewriter.modifyOpInPlace(pool, [&]() {
      pool->setOperand(0, pad.getSource());
    });
    return success();
  }
};

class DropUnreadPadding
    : public impl::DropUnreadPaddingBase<DropUnreadPadding> {
public:
  using impl::DropUnreadPaddingBase<DropUnreadPadding>::DropUnreadPaddingBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<DropPaddingNoPoolReads>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
