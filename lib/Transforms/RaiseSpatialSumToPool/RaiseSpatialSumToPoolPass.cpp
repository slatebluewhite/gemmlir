//===- RaiseSpatialSumToPoolPass.cpp -----------------------------*- C++ -*-===//
//
// `x.mean(dim=(2, 3))` is a global average pool spelled as a bare reduction.
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

#define GEN_PASS_DEF_RAISESPATIALSUMTOPOOL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// A sum over both spatial axes of an image is a global pool, and the rest of
/// the pipeline already knows what to do with one -- but only when it is
/// spelled as `linalg.pooling_n*_sum`.
///
/// `nn.AdaptiveAvgPool2d(1)` gives that spelling. **`x.mean(dim=(2, 3))` does
/// not**: it arrives as a bare `linalg.generic` with two reduction iterators
/// and a separate divide, and nothing downstream matches it. The two are the
/// same operation and the second is how most PyTorch code writes it.
///
/// What it costs to leave alone, measured on `stem`: the convolution above it
/// has no i8 result to end in, so it stays a scalar loop *and*
/// `--conv-to-img2col` reads the i8 four steps below as proof that it folds;
/// and the layout rewrite cannot push NHWC through the reduction, so it leaves
/// a `linalg.transpose` over the whole activation as well.
///
/// Run **before `--conv-nchw-to-nhwc`**: as a pool the layout rewrite carries it
/// along with everything else, and the transpose never appears.
class SpatialSumIsAPool : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.hasPureTensorSemantics())
      return failure();
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();

    auto inTy = llvm::dyn_cast<RankedTensorType>(generic.getInputs()[0].getType());
    auto outTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return failure();
    if (inTy.getRank() != 4 || outTy.getRank() != 2 || !inTy.getElementType().isF32())
      return failure();

    SmallVector<utils::IteratorType> iters = generic.getIteratorTypesArray();
    if (iters.size() != 4)
      return failure();
    SmallVector<unsigned> reduced, kept;
    for (unsigned d = 0; d < 4; d++)
      (iters[d] == utils::IteratorType::reduction ? reduced : kept).push_back(d);
    if (reduced.size() != 2 || reduced[1] != reduced[0] + 1)
      return failure();

    // The two layouts a frontend can hand over. Which axes are summed says
    // which one this is; nothing else about the operation distinguishes them.
    bool nchw = reduced[0] == 2;
    if (!nchw && reduced[0] != 1)
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[0].isIdentity())
      return failure();
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> keptExprs;
    for (unsigned d : kept)
      keptExprs.push_back(getAffineDimExpr(d, ctx));
    if (maps[1] != AffineMap::get(4, 0, keptExprs, ctx))
      return failure();

    // `out = out + in`, and nothing else. An accumulation that also scales or
    // clamps is a different operation and keeps its loop.
    Block &body = generic.getRegion().front();
    if (!llvm::hasSingleElement(body.without_terminator()))
      return failure();
    auto add = llvm::dyn_cast<arith::AddFOp>(&body.front());
    auto yield = llvm::cast<linalg::YieldOp>(body.getTerminator());
    if (!add || yield.getNumOperands() != 1 || yield.getOperand(0) != add.getResult())
      return failure();
    Value a = add.getLhs(), b = add.getRhs();
    if (!((a == body.getArgument(0) && b == body.getArgument(1)) ||
          (a == body.getArgument(1) && b == body.getArgument(0))))
      return failure();

    // The accumulator has to start at zero, or the sum is not the sum.
    auto fill = generic.getOutputs()[0].getDefiningOp<linalg::FillOp>();
    if (!fill || fill.getInputs().size() != 1)
      return failure();
    llvm::APFloat zero(0.0f);
    if (!matchPattern(fill.getInputs()[0], m_ConstantFloat(&zero)) || !zero.isZero() ||
        zero.isNegative())
      return failure();

    int64_t n = inTy.getDimSize(0);
    int64_t h = inTy.getDimSize(reduced[0]), w = inTy.getDimSize(reduced[1]);
    int64_t c = inTy.getDimSize(nchw ? 1 : 3);
    if (outTy.getDimSize(0) != n || outTy.getDimSize(1) != c)
      return failure();

    Location loc = generic.getLoc();
    Type elem = inTy.getElementType();
    SmallVector<int64_t> pooledShape =
        nchw ? SmallVector<int64_t>{n, c, 1, 1} : SmallVector<int64_t>{n, 1, 1, c};

    Value init = rewriter.create<tensor::EmptyOp>(loc, pooledShape, elem);
    Value zeroCst = rewriter.create<arith::ConstantOp>(
        loc, elem, rewriter.getFloatAttr(elem, 0.0));
    Value filled =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zeroCst}, ValueRange{init})
            .getResult(0);
    // The window operand carries no values -- its *shape* is the window.
    Value window = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{h, w}, elem);

    auto ones = rewriter.getI64VectorAttr({1, 1});
    Value pooled;
    if (nchw)
      pooled = rewriter
                   .create<linalg::PoolingNchwSumOp>(
                       loc, TypeRange{filled.getType()},
                       ValueRange{generic.getInputs()[0], window},
                       ValueRange{filled}, ones, ones)
                   .getResult(0);
    else
      pooled = rewriter
                   .create<linalg::PoolingNhwcSumOp>(
                       loc, TypeRange{filled.getType()},
                       ValueRange{generic.getInputs()[0], window},
                       ValueRange{filled}, ones, ones)
                   .getResult(0);

    // (N, C, 1, 1) or (N, 1, 1, C) back to the (N, C) the reduction produced.
    SmallVector<ReassociationIndices> reassoc =
        nchw ? SmallVector<ReassociationIndices>{{0}, {1, 2, 3}}
             : SmallVector<ReassociationIndices>{{0, 1, 2}, {3}};
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(generic, outTy, pooled,
                                                         reassoc);
    return success();
  }
};

class RaiseSpatialSumToPool
    : public impl::RaiseSpatialSumToPoolBase<RaiseSpatialSumToPool> {
public:
  using impl::RaiseSpatialSumToPoolBase<
      RaiseSpatialSumToPool>::RaiseSpatialSumToPoolBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SpatialSumIsAPool>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
