//===- FoldBatchNormPass.cpp -----------------------------------*- C++ -*-===//
//
// Folds a per-channel affine into the weights of the contraction above it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

#include <cmath>

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FOLDBATCHNORM
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// Where the output channel sits in a contraction's result and in its weights.
struct ChannelAxes {
  unsigned result;
  unsigned weights;
};

std::optional<ChannelAxes> channelAxesOf(Operation *op) {
  // linalg counts a filter as FCHW, HWCF, CHW or HWC; the matmul's weights are
  // K x N. In each case the output channel is the one the result is indexed by.
  if (isa<linalg::Conv2DNchwFchwOp>(op))
    return ChannelAxes{1, 0};
  if (isa<linalg::Conv2DNhwcHwcfOp>(op))
    return ChannelAxes{3, 3};
  if (isa<linalg::DepthwiseConv2DNchwChwOp>(op))
    return ChannelAxes{1, 0};
  if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(op))
    return ChannelAxes{3, 2};
  if (isa<linalg::MatmulOp>(op))
    return ChannelAxes{1, 1};
  return std::nullopt;
}

/// A value in the body: either a number, or `a * in + b` in the activation.
struct Linear {
  double a = 0.0, b = 0.0;
  bool isConst() const { return a == 0.0; }
};

/// Evaluates the body for one channel, carrying the activation along as the
/// linear form `1 * in + 0`. Anything that would make the result non-affine --
/// a product of two values that both depend on the activation, a division by
/// one, a square root of one -- fails, and so does an operation not listed
/// here: this is the whole vocabulary it claims to understand.
bool evaluate(Block &body, ArrayRef<double> operands, double *a, double *b) {
  DenseMap<Value, Linear> values;
  values[body.getArgument(0)] = Linear{1.0, 0.0};
  for (unsigned i = 0; i < operands.size(); i++)
    values[body.getArgument(i + 1)] = Linear{0.0, operands[i]};

  auto get = [&](Value v, Linear *out) {
    auto it = values.find(v);
    if (it != values.end()) {
      *out = it->second;
      return true;
    }
    APFloat f(0.0);
    if (matchPattern(v, m_ConstantFloat(&f))) {
      *out = Linear{0.0, f.convertToDouble()};
      return true;
    }
    return false;
  };

  for (Operation &op : body.without_terminator()) {
    if (isa<arith::ConstantOp>(&op))
      continue;
    Linear x, y;
    if (op.getNumOperands() >= 1 && !get(op.getOperand(0), &x))
      return false;
    if (op.getNumOperands() == 2 && !get(op.getOperand(1), &y))
      return false;
    if (op.getNumOperands() > 2 || op.getNumResults() != 1)
      return false;

    Linear r;
    if (isa<arith::AddFOp>(&op))
      r = Linear{x.a + y.a, x.b + y.b};
    else if (isa<arith::SubFOp>(&op))
      r = Linear{x.a - y.a, x.b - y.b};
    else if (isa<arith::NegFOp>(&op))
      r = Linear{-x.a, -x.b};
    else if (isa<arith::TruncFOp, arith::ExtFOp>(&op))
      r = x;
    else if (isa<arith::MulFOp>(&op)) {
      if (!x.isConst() && !y.isConst())
        return false;
      r = x.isConst() ? Linear{y.a * x.b, y.b * x.b} : Linear{x.a * y.b, x.b * y.b};
    } else if (isa<arith::DivFOp>(&op)) {
      if (!y.isConst() || y.b == 0.0)
        return false;
      r = Linear{x.a / y.b, x.b / y.b};
    } else if (isa<math::RsqrtOp>(&op)) {
      if (!x.isConst() || !(x.b > 0.0))
        return false;
      r = Linear{0.0, 1.0 / std::sqrt(x.b)};
    } else if (isa<math::SqrtOp>(&op)) {
      if (!x.isConst() || !(x.b >= 0.0))
        return false;
      r = Linear{0.0, std::sqrt(x.b)};
    } else {
      return false;
    }
    values[op.getResult(0)] = r;
  }

  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  Linear out;
  if (!get(yield.getOperand(0), &out))
    return false;
  *a = out.a;
  *b = out.b;
  return std::isfinite(*a) && std::isfinite(*b);
}

/// The transpose an elementwise operation is carrying in its read map, as
/// `linalg.transpose` counts one. Empty means it reads in order.
///
/// A frontend writes a constant 0 rather than the dimension wherever an axis
/// has extent 1, and that axis then has no dimension of its own in the map --
/// the iteration dimension it stands for is whichever one is left over, and it
/// has extent 1 too, so which way round they go does not matter.
bool relayoutOf(AffineMap read, RankedTensorType activation,
                SmallVectorImpl<int64_t> *permutation) {
  unsigned rank = read.getNumResults();
  if (rank != read.getNumDims() || rank != activation.getRank())
    return false;
  SmallVector<int64_t> reads(rank, -1);
  SmallVector<bool> used(rank, false);
  SmallVector<unsigned> pending;
  for (auto [k, e] : llvm::enumerate(read.getResults())) {
    if (auto dim = dyn_cast<AffineDimExpr>(e)) {
      reads[k] = dim.getPosition();
      used[dim.getPosition()] = true;
      continue;
    }
    auto cst = dyn_cast<AffineConstantExpr>(e);
    if (!cst || cst.getValue() != 0 || activation.getDimSize(k) != 1)
      return false;
    pending.push_back(k);
  }
  for (unsigned d = 0; d < rank && !pending.empty(); d++) {
    if (used[d])
      continue;
    reads[pending.pop_back_val()] = d;
    used[d] = true;
  }
  if (!pending.empty())
    return false;

  // `reads[k]` is the iteration dimension the activation's dimension k is
  // indexed by; `linalg.transpose` wants the other direction.
  permutation->assign(rank, 0);
  for (auto [k, d] : llvm::enumerate(reads))
    (*permutation)[d] = k;
  return true;
}

/// A batch norm in evaluation mode is `out[n, c, ...] = in[n, c, ...] * a[c] +
/// b[c]`, and every term of it is a constant the frontend hands over. Nothing
/// downstream can use it -- `tiled_conv_auto` scales its accumulator by one
/// number, not one per channel -- so a convolution followed by one stays a
/// scalar loop. Scaling output channel `f` of the filter by `a[f]` scales that
/// channel's whole result, so all of it goes into the weights and what is left
/// is an ordinary convolution.
class FoldIntoWeights : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getInputs().empty() || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();

    Value activation = generic.getInputs()[0];
    Operation *producer = activation.getDefiningOp();
    if (!producer)
      return failure();
    std::optional<ChannelAxes> axes = channelAxesOf(producer);
    if (!axes)
      return failure();
    auto contraction = cast<linalg::LinalgOp>(producer);
    if (contraction.getDpsInputs().size() != 2 ||
        contraction->getNumResults() != 1)
      return failure();

    // The result is this operation's to take over: it may feed the destination
    // as well as the input, but nothing else, and the body must not read what
    // was there.
    for (OpOperand &use : activation.getUses())
      if (use.getOwner() != generic)
        return failure();
    Block &body = generic.getRegion().front();
    if (!body.getArguments().back().use_empty())
      return failure();

    auto resTy = dyn_cast<RankedTensorType>(activation.getType());
    if (!resTy || !resTy.hasStaticShape() || !resTy.getElementType().isF32())
      return failure();
    if (resTy.getRank() <= static_cast<int64_t>(axes->result))
      return failure();
    int64_t channels = resTy.getDimSize(axes->result);

    // Every operand after the activation is a per-channel constant, read at the
    // same channel the activation's own index gives.
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != generic.getInputs().size() + 1)
      return failure();
    if (maps[0].getNumResults() != resTy.getRank() ||
        !maps.back().isIdentity())
      return failure();
    AffineExpr channelExpr = maps[0].getResult(axes->result);
    SmallVector<DenseElementsAttr> perChannel;
    for (unsigned i = 1; i < generic.getInputs().size(); i++) {
      auto ty = dyn_cast<RankedTensorType>(generic.getInputs()[i].getType());
      if (!ty || ty.getRank() != 1 || ty.getDimSize(0) != channels)
        return failure();
      if (maps[i].getNumResults() != 1 || maps[i].getResult(0) != channelExpr)
        return failure();
      DenseElementsAttr values;
      if (!matchPattern(generic.getInputs()[i], m_Constant(&values)) ||
          !values.getElementType().isF32())
        return failure();
      perChannel.push_back(values);
    }

    // The affine the body computes, one channel at a time.
    SmallVector<double> scale(channels), offset(channels);
    for (int64_t c = 0; c < channels; c++) {
      SmallVector<double> operands;
      for (DenseElementsAttr attr : perChannel)
        operands.push_back(
            attr.getValues<APFloat>()[c].convertToDouble());
      if (!evaluate(body, operands, &scale[c], &offset[c]))
        return failure();
    }

    // A scale of one is not a scale: a plain bias after a contraction is
    // already something the pipeline reads, and rewriting the weights for it
    // would churn the IR to no purpose.
    if (llvm::all_of(scale, [](double a) { return a == 1.0; }))
      return failure();

    // The weights, scaled by their own output channel.
    DenseElementsAttr weights;
    if (!matchPattern(contraction.getDpsInputs()[1], m_Constant(&weights)) ||
        !weights.getElementType().isF32())
      return failure();
    auto weightTy = cast<RankedTensorType>(contraction.getDpsInputs()[1].getType());
    if (weightTy.getRank() <= static_cast<int64_t>(axes->weights) ||
        weightTy.getDimSize(axes->weights) != channels)
      return failure();
    int64_t inner = 1;
    for (int64_t d = axes->weights + 1; d < weightTy.getRank(); d++)
      inner *= weightTy.getDimSize(d);

    SmallVector<APFloat> scaled;
    scaled.reserve(weightTy.getNumElements());
    int64_t i = 0;
    for (APFloat w : weights.getValues<APFloat>()) {
      int64_t c = (i / inner) % channels;
      scaled.push_back(APFloat(static_cast<float>(w.convertToDouble() * scale[c])));
      i++;
    }

    // Whatever the contraction accumulated onto goes through the same affine.
    SmallVector<double> base(channels, 0.0);
    Value init = contraction.getDpsInits()[0];
    // A grouped convolution's groups accumulate onto slices of one buffer.
    // Only a uniform fill is followed through the slice: every channel of it
    // holds the same number, so which slice this is does not matter. A
    // per-channel broadcast would have to be cut at the right offset, and no
    // frontend has handed one over yet.
    if (auto slice = init.getDefiningOp<tensor::ExtractSliceOp>())
      if (slice.getSource().getDefiningOp<linalg::FillOp>())
        init = slice.getSource();
    if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
      APFloat k(0.0);
      if (fill.getInputs().size() != 1 ||
          !matchPattern(fill.getInputs()[0], m_ConstantFloat(&k)))
        return failure();
      base.assign(channels, k.convertToDouble());
    } else if (auto broadcast = init.getDefiningOp<linalg::GenericOp>()) {
      DenseElementsAttr bias;
      SmallVector<AffineMap> bmaps = broadcast.getIndexingMapsArray();
      auto yield = dyn_cast<linalg::YieldOp>(
          broadcast.getRegion().front().getTerminator());
      if (broadcast.getInputs().size() != 1 || bmaps.size() != 2 ||
          !bmaps.back().isIdentity() || bmaps[0].getNumResults() != 1 ||
          bmaps[0].getResult(0) !=
              getAffineDimExpr(axes->result, rewriter.getContext()) ||
          !yield || yield.getOperand(0) != broadcast.getRegion().front().getArgument(0) ||
          !matchPattern(broadcast.getInputs()[0], m_Constant(&bias)) ||
          !bias.getElementType().isF32() || bias.getNumElements() != channels)
        return failure();
      int64_t c = 0;
      for (APFloat v : bias.getValues<APFloat>())
        base[c++] = v.convertToDouble();
    } else {
      return failure();
    }

    SmallVector<APFloat> newBias;
    for (int64_t c = 0; c < channels; c++)
      newBias.push_back(
          APFloat(static_cast<float>(base[c] * scale[c] + offset[c])));

    // The operation being removed may have been carrying a relayout: the layout
    // rewrite absorbs a convolution's back-transpose into whatever elementwise
    // operation follows, and here that is this one. Work out the transpose to
    // put back before touching anything -- a pattern that gives up after it has
    // already rewritten leaves the driver rewriting the same thing for ever,
    // and it does so with no diagnostic at all.
    SmallVector<int64_t> permutation;
    if (!maps[0].isIdentity() && !relayoutOf(maps[0], resTy, &permutation))
      return failure();

    Location loc = generic.getLoc();
    rewriter.setInsertionPoint(producer);
    Value newWeights = rewriter.create<arith::ConstantOp>(
        loc, weightTy, DenseElementsAttr::get(weightTy, scaled));
    // The same value for every channel is a fill, not a broadcast -- and it is
    // what an average pool leaves, whose divide this folds into an all-ones
    // filter with nothing to add afterwards. Writing it as a broadcast would
    // hide the zero from `--force-quantized-matmul`, which then dequantizes and
    // adds it back: a stray `+ 0.0` in the middle of the requantization, and
    // the requantization matcher does not walk an add.
    Value empty = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                   resTy.getElementType());
    if (llvm::all_of(newBias, [&](const APFloat &v) {
          return v.bitwiseIsEqual(newBias.front());
        })) {
      Value k = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(resTy.getElementType(),
                                     newBias.front().convertToDouble()));
      Value filled =
          rewriter.create<linalg::FillOp>(loc, k, empty).getResult(0);
      rewriter.modifyOpInPlace(producer, [&] {
        producer->setOperand(1, newWeights);
        producer->setOperand(2, filled);
      });
      return finish(rewriter, generic, activation, permutation, loc);
    }

    auto biasTy = RankedTensorType::get({channels}, resTy.getElementType());
    Value biasValues = rewriter.create<arith::ConstantOp>(
        loc, biasTy, DenseElementsAttr::get(biasTy, newBias));

    // The bias joins the accumulator the way a frontend puts it there.
    SmallVector<AffineMap> biasMaps = {
        AffineMap::get(resTy.getRank(), 0,
                       {getAffineDimExpr(axes->result, rewriter.getContext())},
                       rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(resTy.getRank(), rewriter.getContext())};
    SmallVector<utils::IteratorType> iters(resTy.getRank(),
                                           utils::IteratorType::parallel);
    Value newInit =
        rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange{resTy}, ValueRange{biasValues}, ValueRange{empty},
                biasMaps, iters,
                [](OpBuilder &b, Location l, ValueRange args) {
                  b.create<linalg::YieldOp>(l, args[0]);
                })
            .getResult(0);

    rewriter.modifyOpInPlace(producer, [&] {
      producer->setOperand(1, newWeights);
      producer->setOperand(2, newInit);
    });
    return finish(rewriter, generic, activation, permutation, loc);
  }

private:
  /// Puts back whatever relayout the operation being removed was carrying.
  static LogicalResult finish(PatternRewriter &rewriter, linalg::GenericOp generic,
                              Value activation, ArrayRef<int64_t> permutation,
                              Location loc) {
    // The operation being removed may have been carrying a relayout: the layout
    // rewrite absorbs a convolution's back-transpose into whatever elementwise
    // operation follows, and here that was this one. Give the transpose back
    // rather than dropping it -- the absorb patterns downstream will find it
    // another host.
    if (permutation.empty()) {
      rewriter.replaceOp(generic, activation);
      return success();
    }
    auto genericTy = cast<RankedTensorType>(generic->getResult(0).getType());
    // Back down where the operation being replaced was: this one reads the
    // contraction's result, which the constants above it do not.
    rewriter.setInsertionPoint(generic);
    Value dest = rewriter.create<tensor::EmptyOp>(loc, genericTy.getShape(),
                                                  genericTy.getElementType());
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(generic, activation, dest,
                                                     permutation);
    return success();
  }
};

/// **Tried and reverted: a per-channel affine with nothing to fold into.**
///
/// `FoldIntoWeights` needs a contraction above it, and `DistributeOverConcat`
/// needs the join's pieces to be contractions. Fifty-eight of `densenet121`'s
/// 121 batch norms have neither -- they sit on a join of everything the block
/// has produced so far, mostly earlier joins -- and they run
/// `rsqrt(var[c] + eps)` **per element**, 850 thousand of them where one per
/// channel would do. Everything but `x` is a compile-time constant, so the body
/// is `x * A[c] + B[c]` and both can be worked out here.
///
/// It does exactly that and it is a loss. Every `math.rsqrt` in the model goes
/// (597 to 0) and the model gets **worse**: the accelerator's share falls from
/// 514,024 elements to 411,624 and the scalar convolutions go from **1 to 62**.
///
/// The reason is that the batch norm is not a separate operation by then.
/// `--fuse-elementwise-around-matmul` has already put it in one region with the
/// quantization below it, and `matchRequantize` folds that whole region into
/// the convolution's `mvout`. Rewriting the region into a plain affine changes
/// the shape it matches on, and the layer stops folding. Refusing when the
/// producer is a contraction does not help, because the producer here *is* the
/// join -- the damage is to the convolution below.
///
/// What would settle it is doing this where the tail is still separate, before
/// the fusion; at that point, though, the constants have not been folded and
/// there is nothing to evaluate. The `rsqrt` per element is real and still
/// there.

/// A per-channel affine over a `tensor.concat`, pushed into the pieces.
///
/// `--split-grouped-conv` writes a grouped convolution as G convolutions joined
/// by a concatenation, and the batch norm the frontend emitted sits on the
/// join. `FoldIntoWeights` then finds a concatenation above it rather than a
/// contraction and gives up, so not one of the G convolutions ever loses its
/// batch norm -- which is what kept every model in the grouped family at one
/// offloaded operation between them.
///
/// Each output channel belongs to exactly one group, so the per-channel
/// constants cut along the joined axis and each piece's share folds into its
/// own group's weights.
class DistributeOverConcat : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getInputs().size() < 2 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    Block &body = generic.getRegion().front();
    if (!body.getArguments().back().use_empty())
      return failure();

    auto concat = generic.getInputs()[0].getDefiningOp<tensor::ConcatOp>();
    if (!concat || !concat->hasOneUse() || concat.getInputs().size() < 2)
      return failure();
    auto srcTy = dyn_cast<RankedTensorType>(concat.getType());
    auto resTy = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!srcTy || !resTy || !srcTy.hasStaticShape() || !resTy.hasStaticShape() ||
        !resTy.getElementType().isF32())
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != generic.getInputs().size() + 1 ||
        !maps.back().isIdentity() ||
        maps[0].getNumResults() != srcTy.getRank())
      return failure();

    // Which iteration dimension walks the joined axis: the per-channel
    // constants have to be read at exactly that one.
    auto joined = dyn_cast<AffineDimExpr>(maps[0].getResult(concat.getDim()));
    if (!joined)
      return failure();

    // Everything after the activation is a per-channel constant read at that
    // axis, which is what makes the cut possible.
    SmallVector<DenseElementsAttr> perChannel;
    for (unsigned i = 1; i < generic.getInputs().size(); i++) {
      auto ty = dyn_cast<RankedTensorType>(generic.getInputs()[i].getType());
      if (!ty || ty.getRank() != 1 ||
          ty.getDimSize(0) != srcTy.getDimSize(concat.getDim()))
        return failure();
      if (maps[i].getNumResults() != 1 || maps[i].getResult(0) != joined)
        return failure();
      DenseElementsAttr values;
      if (!matchPattern(generic.getInputs()[i], m_Constant(&values)) ||
          !values.getElementType().isF32())
        return failure();
      perChannel.push_back(values);
    }

    // The pieces are built in the concatenation's own layout and joined on its
    // own axis; whatever relayout the operation was carrying stays behind as a
    // single copy over the join. Giving each group its own transpose instead
    // would leave G of them between the convolutions and the requantization,
    // which is one more thing for it to be pushed back through.
    Location loc = generic.getLoc();
    Type elem = resTy.getElementType();
    MLIRContext *ctx = rewriter.getContext();
    unsigned rank = srcTy.getRank();
    AffineMap ident = rewriter.getMultiDimIdentityMap(rank);
    AffineMap channel = AffineMap::get(
        rank, 0, {getAffineDimExpr(concat.getDim(), ctx)}, ctx);
    SmallVector<AffineMap> pieceMaps{ident};
    pieceMaps.append(perChannel.size(), channel);
    pieceMaps.push_back(ident);
    SmallVector<utils::IteratorType> parallel(rank,
                                              utils::IteratorType::parallel);

    SmallVector<Value> pieces;
    int64_t at = 0;
    for (Value piece : concat.getInputs()) {
      auto pieceTy = dyn_cast<RankedTensorType>(piece.getType());
      if (!pieceTy || !pieceTy.hasStaticShape() || pieceTy.getRank() != rank)
        return failure();
      int64_t width = pieceTy.getDimSize(concat.getDim());

      SmallVector<Value> ins{piece};
      auto cutTy = RankedTensorType::get({width}, elem);
      for (DenseElementsAttr whole : perChannel) {
        SmallVector<APFloat> cut;
        int64_t c = 0;
        for (APFloat v : whole.getValues<APFloat>()) {
          if (c >= at && c < at + width)
            cut.push_back(v);
          c++;
        }
        ins.push_back(rewriter.create<arith::ConstantOp>(
            loc, cutTy, DenseElementsAttr::get(cutTy, cut)));
      }

      auto outTy = RankedTensorType::get(pieceTy.getShape(), elem);
      Value init =
          rewriter.create<tensor::EmptyOp>(loc, pieceTy.getShape(), elem);
      auto one = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{outTy}, ins, ValueRange{init}, pieceMaps, parallel);
      rewriter.cloneRegionBefore(generic.getRegion(), one.getRegion(),
                                 one.getRegion().begin());
      pieces.push_back(one.getResult(0));
      at += width;
    }

    Value joinTy = rewriter.create<tensor::ConcatOp>(
        loc, concat.getDim(), pieces);
    Value dest = rewriter.create<tensor::EmptyOp>(loc, resTy, ValueRange{});
    auto relayout = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{joinTy}, ValueRange{dest},
        SmallVector<AffineMap>{maps[0], maps.back()},
        generic.getIteratorTypesArray(),
        [](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, args[0]);
        });
    rewriter.replaceOp(generic, relayout.getResults());
    return success();

  }
};

class FoldBatchNorm : public impl::FoldBatchNormBase<FoldBatchNorm> {
public:
  using impl::FoldBatchNormBase<FoldBatchNorm>::FoldBatchNormBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldIntoWeights, DistributeOverConcat>(&getContext());
    // A grouped convolution's G filters are G slices of one constant, and a
    // slice is not a constant: `FoldIntoWeights` needs to read the weights it
    // is about to scale, and every later pass that reads a weight has the same
    // problem. Materialize them here, once. Only where the slices are all there
    // is of the constant, so what they cost is what it cost and the original
    // dies as soon as the last one folds.
    tensor::populateFoldConstantExtractSlicePatterns(
        patterns, [](tensor::ExtractSliceOp op) {
          return llvm::all_of(op.getSource().getUsers(), [](Operation *user) {
            return llvm::isa<tensor::ExtractSliceOp>(user);
          });
        });
    // Each fold rewrites a filter into a fresh constant and re-points the
    // convolution at it, which puts the convolution and everything downstream
    // of it back on the worklist; a network with a batch norm after every layer
    // needs more than the default ten passes over the region to settle.
    GreedyRewriteConfig config;
    config.setMaxIterations(64);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
