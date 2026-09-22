//===- HoistElementwiseBeforeGatherPass.cpp -----------------*- C++ -*-===//
//
// Moves an elementwise operation to the other side of a pure gather.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_HOISTELEMENTWISEBEFOREGATHER
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The iteration dimension each dimension of a read operand is walked by.
///
/// A frontend writes a constant 0 rather than the dimension wherever an axis
/// has extent 1, so that axis has no dimension of its own in the map; the
/// iteration dimension it stands for is whichever one is left over, and it has
/// extent 1 too, so which way round they go does not matter. `isPermutation()`
/// says no to exactly those maps, and an NCHW-to-NHWC relayout on a batch of
/// one is written that way -- which is what kept the requantization on the far
/// side of every grouped convolution's join.
static bool readOrder(AffineMap read, ArrayRef<int64_t> srcShape,
                      SmallVectorImpl<int64_t> &order) {
  unsigned rank = read.getNumResults();
  if (rank != read.getNumDims() || rank != srcShape.size())
    return false;
  order.assign(rank, -1);
  SmallVector<bool> used(rank, false);
  SmallVector<unsigned> pending;
  for (auto [k, e] : llvm::enumerate(read.getResults())) {
    if (auto dim = llvm::dyn_cast<AffineDimExpr>(e)) {
      order[k] = dim.getPosition();
      used[dim.getPosition()] = true;
      continue;
    }
    auto cst = llvm::dyn_cast<AffineConstantExpr>(e);
    if (!cst || cst.getValue() != 0 || srcShape[k] != 1)
      return false;
    pending.push_back(k);
  }
  for (unsigned d = 0; d < rank && !pending.empty(); d++) {
    if (used[d])
      continue;
    order[pending.pop_back_val()] = d;
    used[d] = true;
  }
  return pending.empty();
}

/// A linalg.generic that only copies: one input, one output, and a body that
/// yields its input unchanged. im2col packing is one of these -- its index map
/// has floordiv and mod in it, which is why the ordinary elementwise fusion
/// cannot touch it.
static bool isPureGather(linalg::GenericOp op) {
  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
    return false;
  if (!llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;
  Block &body = op.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yield && yield.getNumOperands() == 1 &&
         yield.getOperand(0) == body.getArgument(0);
}

/// An elementwise map-in-place: identity maps, one input, one output, and the
/// output not read by the body.
static bool isElementwiseMap(linalg::GenericOp op) {
  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
    return false;
  if (!llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;
  if (!llvm::all_of(op.getIndexingMapsArray(),
                    [](AffineMap m) { return m.isIdentity(); }))
    return false;
  Block &body = op.getRegion().front();
  return body.getArgument(1).use_empty();
}

/// True when the body maps zero to zero, which is what lets an elementwise
/// operation move to the other side of a zero pad.
///
/// Only operations that provably do are followed. A quantization is all of
/// them: dividing by a scale, rounding, converting, and clamping to a range
/// that contains zero all leave zero where it is.
static bool mapsZeroToZero(Value v, BlockArgument arg, unsigned depth = 0) {
  if (v == arg)
    return true;
  if (depth > 32)
    return false;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  auto through = [&](Value x) { return mapsZeroToZero(x, arg, depth + 1); };
  auto constant = [](Value c, double *out) {
    llvm::APFloat f(0.0f);
    if (matchPattern(c, m_ConstantFloat(&f))) {
      *out = f.convertToDouble();
      return true;
    }
    llvm::APInt i;
    if (matchPattern(c, m_ConstantInt(&i))) {
      *out = (double)i.getSExtValue();
      return true;
    }
    return false;
  };

  if (llvm::isa<math::RoundEvenOp, arith::FPToSIOp, arith::SIToFPOp,
                arith::ExtSIOp, arith::ExtFOp, arith::TruncIOp,
                arith::TruncFOp, arith::NegFOp>(def))
    return through(def->getOperand(0));
  double c = 0.0;
  if (auto op = llvm::dyn_cast<arith::MulFOp>(def))
    return (constant(op.getRhs(), &c) && through(op.getLhs())) ||
           (constant(op.getLhs(), &c) && through(op.getRhs()));
  if (auto op = llvm::dyn_cast<arith::DivFOp>(def))
    return constant(op.getRhs(), &c) && c != 0.0 && through(op.getLhs());
  // A clamp keeps zero only if zero is inside it.
  if (llvm::isa<arith::MaxSIOp, arith::MaximumFOp, arith::MaxNumFOp>(def))
    return constant(def->getOperand(1), &c) && c <= 0.0 &&
           through(def->getOperand(0));
  if (llvm::isa<arith::MinSIOp, arith::MinimumFOp, arith::MinNumFOp>(def))
    return constant(def->getOperand(1), &c) && c >= 0.0 &&
           through(def->getOperand(0));
  return false;
}

/// `elementwise(pad(x, 0))` is `pad(elementwise(x), 0)` when the operation
/// leaves zero where it is.
///
/// A convolution's padding bufferizes into a fill of the whole padded buffer
/// plus a copy of the real input into the middle of it, and doing that before
/// the quantization means both run on f32. Afterwards they run on i8 -- a
/// quarter of the memory traffic -- and the conversion itself covers only the
/// real input rather than the padded one: 972 elements to 768 on the CNN.
/// True when every element of `v` is known to be at least zero.
///
/// Only the shapes a relu arrives in. `--select-to-minmax` has not run yet at
/// this point in the pipeline, so a frontend's relu is still a compare and a
/// select; after it, it is a max.
static bool provablyNonNegative(Value v, unsigned depth = 0) {
  // GoogLeNet's pool branch sits five steps above its relu -- a transpose, a
  // concatenation, a transpose again -- and an inception module whose pool
  // branch feeds the next one's is deeper still. A short budget here is the
  // mistake [[gemmlir-a-budget-is-not-a-rule]] records, so the walk is bounded
  // only against a cycle; every step below either recurses on one operand or
  // ends.
  if (depth > 12)
    return false;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (auto transpose = llvm::dyn_cast<linalg::TransposeOp>(def))
    return provablyNonNegative(transpose.getInput(), depth + 1);

  // A join is non-negative when every branch is. This is what GoogLeNet needs:
  // each inception module pools the concatenation of the previous one's four
  // branches, and all four end in a relu.
  if (auto concat = llvm::dyn_cast<tensor::ConcatOp>(def))
    return llvm::all_of(concat.getInputs(), [&](Value in) {
      return provablyNonNegative(in, depth + 1);
    });

  // `max(window)` over non-negative elements is non-negative, and so is a
  // padding that is itself non-negative. The two together are how a pool of a
  // pool clears: once this pattern has rewritten the inner padding to zero, the
  // pool above it is provable in turn.
  if (llvm::isa<linalg::PoolingNhwcMaxOp, linalg::PoolingNchwMaxOp>(def))
    return provablyNonNegative(def->getOperand(0), depth + 1);

  if (auto pad = llvm::dyn_cast<tensor::PadOp>(def)) {
    Value padded = pad.getConstantPaddingValue();
    llvm::APFloat f(0.0f);
    if (!padded || !matchPattern(padded, m_ConstantFloat(&f)) || f.isNaN() ||
        f.isNegative())
      return false;
    return provablyNonNegative(pad.getSource(), depth + 1);
  }

  auto generic = llvm::dyn_cast<linalg::GenericOp>(def);
  if (!generic || generic.getNumDpsInits() != 1)
    return false;
  auto yield =
      llvm::dyn_cast<linalg::YieldOp>(generic.getRegion().front().getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;

  auto nonNegativeConstant = [](Value c) {
    llvm::APFloat f(0.0f);
    if (matchPattern(c, m_ConstantFloat(&f)))
      return !f.isNaN() && !f.isNegative();
    llvm::APInt i;
    if (matchPattern(c, m_ConstantInt(&i)))
      return !i.isNegative();
    return false;
  };

  Operation *tail = yield.getOperand(0).getDefiningOp();
  if (!tail)
    return false;
  if (llvm::isa<arith::MaximumFOp, arith::MaxNumFOp, arith::MaxSIOp>(tail))
    return llvm::any_of(tail->getOperands(), nonNegativeConstant);
  // `x > c ? x : c` with `c >= 0`, which is what a relu lowers to before
  // --select-to-minmax rewrites it.
  if (auto select = llvm::dyn_cast<arith::SelectOp>(tail)) {
    auto cmp = select.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!cmp)
      return false;
    bool greater = cmp.getPredicate() == arith::CmpFPredicate::UGT ||
                   cmp.getPredicate() == arith::CmpFPredicate::OGT;
    return greater && cmp.getLhs() == select.getTrueValue() &&
           cmp.getRhs() == select.getFalseValue() &&
           nonNegativeConstant(select.getFalseValue());
  }
  return false;
}

/// A max-pool padded with the type's minimum is padded with **zero** instead,
/// when what it pools cannot be negative.
///
/// Both give `max(window)` -- the padding loses to any real element that is at
/// least zero, and a relu above guarantees every one of them is. It matters
/// because zero is the padding Gemmini's own pooling does: `sp_tiled_conv`'s
/// out-of-bounds branch reads zero, not -inf, so this is what lets a padded
/// max-pool fold into the convolution's `pool_padding` at all. It also keeps a
/// poison out of the IR -- `fptosi(-inf)` is what the padding value became once
/// `HoistElementwiseBeforePad` started moving it through the quantization.
///
/// The one thing to check is that no output window is **entirely** padding,
/// since then there is no real element for the padding to lose to.
/// True when the body reads its own position in the iteration space.
///
/// Every pattern here moves a body into a **different** iteration space -- past
/// a pad, a slice, a transpose, a concatenation, a gather. A `linalg.index` is
/// the one thing in a body whose meaning is the space it sits in, so moving it
/// is wrong even when the arithmetic around it is not. It is also not a
/// theoretical case: an embedding lookup is a `linalg.generic` that reads its
/// index, and a decoder-only transformer starts with one. Left unguarded this
/// crashed MLIR's own folder, which asserts the dimension is in range.
static bool readsItsIndex(Operation *op) {
  bool found = false;
  op->walk([&](linalg::IndexOp) { found = true; });
  return found;
}

class ZeroPadAMaxPool : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp pad,
                                PatternRewriter &rewriter) const final {
    Value padValue = pad.getConstantPaddingValue();
    if (!padValue)
      return failure();
    llvm::APFloat f(0.0f);
    if (!matchPattern(padValue, m_ConstantFloat(&f)) || f.isNaN() ||
        !f.isNegative() || f.isZero())
      return failure();
    if (!pad->hasOneUse())
      return failure();

    // Directly below the pool. `MovePadThroughTranspose` is what makes that
    // true when the layout rewrite has left a relayout in between.
    Operation *user = *pad->getUsers().begin();
    unsigned first;
    if (llvm::isa<linalg::PoolingNhwcMaxOp>(user))
      first = 1;
    else if (llvm::isa<linalg::PoolingNchwMaxOp>(user))
      first = 2;
    else
      return failure();
    auto pool = llvm::cast<linalg::LinalgOp>(user);
    if (pool->getOperand(0) != pad.getResult())
      return failure();
    if (!provablyNonNegative(pad.getSource()))
      return failure();

    auto padTy = llvm::cast<RankedTensorType>(pad.getType());
    auto srcTy = llvm::dyn_cast<RankedTensorType>(pad.getSource().getType());
    auto outTy = llvm::dyn_cast<RankedTensorType>(pool->getResult(0).getType());
    auto windowTy =
        llvm::dyn_cast<RankedTensorType>(pool->getOperand(1).getType());
    if (!srcTy || !outTy || !windowTy || !srcTy.hasStaticShape() ||
        !outTy.hasStaticShape() || !windowTy.hasStaticShape() ||
        !padTy.hasStaticShape() || windowTy.getRank() != 2)
      return failure();

    ArrayRef<int64_t> low = pad.getStaticLow(), high = pad.getStaticHigh();
    if (low.size() != (size_t)padTy.getRank() || high.size() != low.size() ||
        llvm::any_of(low, ShapedType::isDynamic) ||
        llvm::any_of(high, ShapedType::isDynamic))
      return failure();
    for (unsigned d = 0; d < low.size(); d++) {
      bool spatial = d == first || d == first + 1;
      if (!spatial && (low[d] != 0 || high[d] != 0))
        return failure();
    }

    auto pair = [](DenseIntElementsAttr a, unsigned k) -> std::optional<int64_t> {
      if (!a || a.getNumElements() != 2)
        return std::nullopt;
      return (*(a.value_begin<APInt>() + k)).getSExtValue();
    };
    for (unsigned k = 0; k < 2; k++) {
      unsigned d = first + k;
      std::optional<int64_t> stride =
          pair(pool->getAttrOfType<DenseIntElementsAttr>("strides"), k);
      std::optional<int64_t> dilation =
          pair(pool->getAttrOfType<DenseIntElementsAttr>("dilations"), k);
      if (!stride || !dilation || *stride < 1 || *dilation < 1)
        return failure();
      int64_t span = (windowTy.getShape()[k] - 1) * *dilation;
      int64_t last = (outTy.getShape()[d] - 1) * *stride;
      // The first window has to reach the image, and the last has to start
      // before it ends.
      if (span < low[d] || last > low[d] + srcTy.getShape()[d] - 1)
        return failure();
    }

    Value zero = rewriter.create<arith::ConstantOp>(
        pad.getLoc(), padValue.getType(),
        rewriter.getZeroAttr(padValue.getType()));
    rewriter.replaceOpWithNewOp<tensor::PadOp>(
        pad, padTy, pad.getSource(), pad.getMixedLowPad(),
        pad.getMixedHighPad(), zero, /*nofold=*/false);
    return success();
  }
};

/// `transpose(pad(x, v))` is `pad(transpose(x), v)` with the padding permuted
/// the same way. Exact -- a padding writes `v` or copies `x`, and a relayout
/// does not care which.
///
/// It is here because it is what stands between a padded max-pool and
/// `HoistElementwiseBeforePad` below. The layout rewrite turns the pool into
/// NHWC but leaves the frontend's pad in NCHW, so the order is
/// `pad -> transpose -> quantize` and the quantization never sees the padding
/// it could move across. Moving the pad down puts them next to each other, and
/// shrinks the relayout to the unpadded image on the way.
class MovePadThroughTranspose : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    if (!transpose.hasPureTensorSemantics())
      return failure();
    auto pad = transpose.getInput().getDefiningOp<tensor::PadOp>();
    if (!pad || !pad->hasOneUse())
      return failure();
    Value padValue = pad.getConstantPaddingValue();
    if (!padValue)
      return failure();

    auto srcTy = llvm::dyn_cast<RankedTensorType>(pad.getSource().getType());
    auto resTy =
        llvm::dyn_cast<RankedTensorType>(transpose.getResult()[0].getType());
    if (!srcTy || !resTy || !srcTy.hasStaticShape() || !resTy.hasStaticShape())
      return failure();
    ArrayRef<int64_t> perm = transpose.getPermutation();
    unsigned rank = srcTy.getRank();
    if (perm.size() != rank)
      return failure();

    // `linalg.transpose` gives `dim(result, k) == dim(input, perm[k])`, so
    // result dimension k carries the source dimension -- and the padding -- of
    // input dimension perm[k].
    Location loc = transpose.getLoc();
    SmallVector<int64_t> movedShape(rank);
    for (unsigned k = 0; k < rank; k++)
      movedShape[k] = srcTy.getShape()[perm[k]];
    Value init = rewriter.create<tensor::EmptyOp>(loc, movedShape,
                                                  srcTy.getElementType());
    Value moved = rewriter
                      .create<linalg::TransposeOp>(loc, pad.getSource(), init,
                                                   perm)
                      .getResult()[0];

    SmallVector<OpFoldResult> lowIn = pad.getMixedLowPad();
    SmallVector<OpFoldResult> highIn = pad.getMixedHighPad();
    SmallVector<OpFoldResult> low(rank), high(rank);
    for (unsigned k = 0; k < rank; k++) {
      low[k] = lowIn[perm[k]];
      high[k] = highIn[perm[k]];
    }
    rewriter.replaceOpWithNewOp<tensor::PadOp>(transpose, resTy, moved, low,
                                               high, padValue,
                                               /*nofold=*/false);
    return success();
  }
};

class HoistElementwiseBeforePad : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (readsItsIndex(generic))
      return failure();
    // Like isElementwiseMap, except that the read may be permuted: by this
    // point the conversion has usually absorbed the layout rewrite's transpose,
    // and the padding simply moves with it.
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity())
      return failure();
    if (!generic.getRegion().front().getArgument(1).use_empty())
      return failure();
    auto pad = generic.getInputs()[0].getDefiningOp<tensor::PadOp>();
    if (!pad || !pad->hasOneUse())
      return failure();
    Value padValue = pad.getConstantPaddingValue();
    if (!padValue)
      return failure();
    // Zero is the common case and gets the cheap answer below; anything else
    // has to be put through the operation being moved.
    bool zeroPad = matchPattern(padValue, m_AnyZeroFloat());

    auto srcTy = llvm::dyn_cast<RankedTensorType>(pad.getSource().getType());
    auto resTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!srcTy || !resTy || !srcTy.hasStaticShape() || !resTy.hasStaticShape())
      return failure();

    // Only in the narrowing direction. The point is that the fill and the copy
    // a padding bufferizes into should run on the *smaller* type; moving a
    // dequantization the same way makes the padded buffer four times the bytes
    // instead. `grp` did exactly that once a grouped convolution's slices
    // started moving above their shared dequantization: the padding went to
    // f32 and the pointwise convolution above it stopped folding, 19.3 ->
    // 24.8 ms.
    Type srcElem = srcTy.getElementType(), resElem = resTy.getElementType();
    if (!srcElem.isIntOrFloat() || !resElem.isIntOrFloat() ||
        resElem.getIntOrFloatBitWidth() > srcElem.getIntOrFloatBitWidth())
      return failure();

    Block &body = generic.getRegion().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    // `pad(x, v)` put through an elementwise `q` is `pad(q(x), q(v))` -- exactly,
    // for any `q`, because padding only ever writes `v` or copies `x`. For a
    // zero pad and a `q` that keeps zero at zero the new value is a literal
    // zero; otherwise it is `q` itself, run once on the scalar.
    //
    // A max-pool pads with **-inf**, which is the case this exists for: without
    // it the padded buffer stays f32 and four times the bytes, the quantization
    // runs over the padding as well as the image, and -- because the padding
    // sits between the convolution and the requantization -- the convolution
    // above ends in something bigger than it wrote and stops folding.
    bool literalZero =
        zeroPad && mapsZeroToZero(yield.getOperand(0), body.getArgument(0));
    if (!literalZero) {
      if (body.getArgument(0).getType() != padValue.getType())
        return failure();
      for (Operation &op : body.without_terminator())
        if (!isPure(&op) || op.getNumResults() != 1 ||
            op.getNumRegions() != 0 || llvm::isa<linalg::IndexOp>(&op))
          return failure();
    }

    // The read map says which iteration dimension walks each source dimension.
    // That fixes both the shape the moved operation writes and where the
    // padding of each source dimension ends up. `readOrder` rather than
    // `isPermutation`, because an NCHW-to-NHWC relayout on a batch of one is
    // written with a constant 0 for the batch axis -- and that is the shape a
    // ShuffleNet unit's quantization arrives in, sitting on a padded f32 buffer
    // between the shuffle and the convolution that reads it.
    unsigned rank = maps[0].getNumResults();
    auto padTy = llvm::cast<RankedTensorType>(pad.getType());
    SmallVector<int64_t> order;
    if (rank != (unsigned)srcTy.getRank() || rank != (unsigned)resTy.getRank() ||
        !readOrder(maps[0], padTy.getShape(), order))
      return failure();

    Location loc = generic.getLoc();
    Type elem = resTy.getElementType();
    SmallVector<int64_t> earlyShape(rank, 0);
    for (unsigned r = 0; r < rank; r++)
      earlyShape[order[r]] = srcTy.getShape()[r];
    auto earlyTy = RankedTensorType::get(earlyShape, elem);
    Value init = rewriter.create<tensor::EmptyOp>(loc, earlyTy.getShape(), elem);
    auto early = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{earlyTy}, ValueRange{pad.getSource()}, ValueRange{init},
        generic.getIndexingMapsArray(), generic.getIteratorTypesArray());
    rewriter.cloneRegionBefore(generic.getRegion(), early.getRegion(),
                               early.getRegion().begin());

    SmallVector<OpFoldResult> lowIn = pad.getMixedLowPad();
    SmallVector<OpFoldResult> highIn = pad.getMixedHighPad();
    SmallVector<OpFoldResult> low(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> high(rank, rewriter.getIndexAttr(0));
    for (unsigned r = 0; r < rank; r++) {
      low[order[r]] = lowIn[r];
      high[order[r]] = highIn[r];
    }

    Value moved;
    if (literalZero) {
      moved = rewriter.create<arith::ConstantOp>(loc, elem,
                                                 rewriter.getZeroAttr(elem));
    } else {
      IRMapping scalar;
      scalar.map(body.getArgument(0), padValue);
      for (Operation &op : body.without_terminator())
        rewriter.clone(op, scalar);
      moved = scalar.lookupOrDefault(yield.getOperand(0));
    }
    rewriter.replaceOpWithNewOp<tensor::PadOp>(
        generic, resTy, early.getResult(0), low, high, moved, /*nofold=*/false);
    return success();
  }
};

/// A slice of a padded value is the padding of the slice, when the slice takes
/// the whole of every axis the padding touches.
///
/// A grouped convolution pads its input on the two spatial axes and then takes
/// one channel slice per group, and those two commute exactly. Taken in the
/// other order the padding happens on each group's own channels -- and, more to
/// the point, whatever quantization sits on the slice can then move in front of
/// the padding, so the fill and the copy a padding bufferizes into run on i8
/// instead of f32. On `gmid` the padding was 10368 f32 elements filled and an
/// 8192-element f32 copy into the middle of them.
class MoveSliceBeforePad : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp slice,
                                PatternRewriter &rewriter) const final {
    if (Operation *def = slice.getSource().getDefiningOp())
      if (readsItsIndex(def))
        return failure();
    auto pad = slice.getSource().getDefiningOp<tensor::PadOp>();
    if (!pad)
      return failure();
    auto padTy = llvm::dyn_cast<RankedTensorType>(pad.getType());
    auto srcTy = llvm::dyn_cast<RankedTensorType>(pad.getSource().getType());
    auto resTy = llvm::dyn_cast<RankedTensorType>(slice.getType());
    if (!padTy || !srcTy || !resTy || !padTy.hasStaticShape() ||
        !srcTy.hasStaticShape() || !resTy.hasStaticShape())
      return failure();
    Value padValue = pad.getConstantPaddingValue();
    if (!padValue)
      return failure();

    // One use, or a set of slices asking for no more than the whole: the
    // padding is then done once per group instead of once for all of them,
    // which is the same number of elements.
    if (!pad->hasOneUse()) {
      int64_t asked = 0;
      for (Operation *user : pad->getUsers()) {
        auto other = llvm::dyn_cast<tensor::ExtractSliceOp>(user);
        auto ty = other ? llvm::dyn_cast<RankedTensorType>(other.getType())
                        : RankedTensorType();
        if (!ty || !ty.hasStaticShape())
          return failure();
        asked += ty.getNumElements();
      }
      if (asked > padTy.getNumElements())
        return failure();
    }

    unsigned rank = padTy.getRank();
    SmallVector<OpFoldResult> offsets = slice.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = slice.getMixedSizes();
    SmallVector<OpFoldResult> strides = slice.getMixedStrides();
    if (offsets.size() != rank || sizes.size() != rank || strides.size() != rank)
      return failure();
    auto constantOf = [](OpFoldResult v, int64_t *out) {
      auto attr = llvm::dyn_cast<Attribute>(v);
      if (!attr)
        return false;
      *out = llvm::cast<IntegerAttr>(attr).getInt();
      return true;
    };

    SmallVector<OpFoldResult> low = pad.getMixedLowPad();
    SmallVector<OpFoldResult> high = pad.getMixedHighPad();
    for (unsigned d = 0; d < rank; d++) {
      int64_t stride = 0;
      if (!constantOf(strides[d], &stride) || stride != 1)
        return failure();
      int64_t lo = 0, hi = 0;
      if (!constantOf(low[d], &lo) || !constantOf(high[d], &hi))
        return failure();
      if (lo == 0 && hi == 0)
        continue;
      // A padded axis has to be taken whole; anything else would cut into the
      // border and the two operations would not commute.
      int64_t off = 0, size = 0;
      if (!constantOf(offsets[d], &off) || !constantOf(sizes[d], &size))
        return failure();
      if (off != 0 || size != padTy.getDimSize(d))
        return failure();
    }

    // The slice, now on the unpadded source: the axes the padding touches are
    // taken whole there too, and the rest keep their window.
    SmallVector<OpFoldResult> innerOffsets(offsets), innerSizes(sizes);
    SmallVector<int64_t> innerShape(resTy.getShape());
    for (unsigned d = 0; d < rank; d++) {
      int64_t lo = 0, hi = 0;
      constantOf(low[d], &lo);
      constantOf(high[d], &hi);
      if (lo == 0 && hi == 0)
        continue;
      innerOffsets[d] = rewriter.getIndexAttr(0);
      innerSizes[d] = rewriter.getIndexAttr(srcTy.getDimSize(d));
      innerShape[d] = srcTy.getDimSize(d);
    }

    Location loc = slice.getLoc();
    Value inner = rewriter.create<tensor::ExtractSliceOp>(
        loc, RankedTensorType::get(innerShape, srcTy.getElementType()),
        pad.getSource(), innerOffsets, innerSizes, strides);
    auto moved = rewriter.create<tensor::PadOp>(loc, resTy, inner, low, high,
                                                padValue, /*nofold=*/false);
    rewriter.replaceOp(slice, moved.getResult());
    return success();
  }
};

/// A slice of an elementwise result is that operation over the slice.
///
/// Where the channels are split -- ShuffleNet's unit passes half of them
/// through untouched -- `--share-branch-quantization` leaves the other half
/// reading the shared activation back through a dequantization and a relayout,
/// both over the *whole* tensor, before slicing and quantizing again. Taking
/// the slice first does the same work on the half that is wanted: 2048 elements
/// instead of 4096 + 4096 + 2048, and what is left fuses into one pass from i8
/// to i8.
class MoveSliceBeforeElementwise : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp slice,
                                PatternRewriter &rewriter) const final {
    if (Operation *def = slice.getSource().getDefiningOp())
      if (readsItsIndex(def))
        return failure();
    Operation *producer = slice.getSource().getDefiningOp();
    if (!producer)
      return failure();
    // One use, or a set of slices that between them ask for no more than the
    // whole. A grouped convolution is the second: `--share-branch-quantization`
    // leaves one dequantization of the joined activation and one slice per
    // group, so moving every slice above it replaces one pass over the whole
    // with one pass over each part -- and the dequantization then dies. Left
    // alone, `grp` and `gup` each carried a 10368-element dequantization to
    // f32 and four 2592-element quantizations straight back, at the *same*
    // scale.
    if (!producer->hasOneUse()) {
      int64_t whole = 0, asked = 0;
      if (auto ty = llvm::dyn_cast<RankedTensorType>(slice.getSource().getType()))
        whole = ty.getNumElements();
      for (Operation *user : producer->getUsers()) {
        auto other = llvm::dyn_cast<tensor::ExtractSliceOp>(user);
        if (!other)
          return failure();
        auto ty = llvm::dyn_cast<RankedTensorType>(other.getType());
        if (!ty || !ty.hasStaticShape())
          return failure();
        asked += ty.getNumElements();
      }
      if (whole <= 0 || asked > whole)
        return failure();
      // ... and not when what feeds the elementwise is an accelerator layer.
      // Splitting the pass restructures everything between here and there, and
      // where that reaches a convolution the convolution stops folding: `grp`
      // traded a `conv2d_i8` for a `matmul_i8` and went 19.3 -> 24.8 ms, `gup`
      // 33.6 -> 35.4. Where the chain ends at the function's own argument --
      // `gmid`, `gmin`, `gdown` -- there is nothing to break and the round trip
      // is 10368 elements each.
      Value up = producer->getNumOperands() ? producer->getOperand(0) : Value();
      for (unsigned step = 0; up && step < 8; step++) {
        Operation *def = up.getDefiningOp();
        if (!def)
          break;
        if (llvm::isa<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(def))
          return failure();
        if (auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(def))
          if (linalg::isaContractionOpInterface(linalgOp))
            return failure();
        if (auto pad = llvm::dyn_cast<tensor::PadOp>(def)) { up = pad.getSource(); continue; }
        if (auto ex = llvm::dyn_cast<tensor::ExpandShapeOp>(def)) { up = ex.getSrc(); continue; }
        if (auto co = llvm::dyn_cast<tensor::CollapseShapeOp>(def)) { up = co.getSrc(); continue; }
        if (auto g = llvm::dyn_cast<linalg::GenericOp>(def)) {
          if (g.getInputs().empty())
            break;
          up = g.getInputs()[0];
          continue;
        }
        break;
      }
    }
    auto resTy = llvm::dyn_cast<RankedTensorType>(slice.getType());
    auto srcTy = llvm::dyn_cast<RankedTensorType>(slice.getSource().getType());
    if (!resTy || !srcTy || resTy.getRank() != srcTy.getRank())
      return failure();
    unsigned rank = srcTy.getRank();

    SmallVector<OpFoldResult> offsets = slice.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = slice.getMixedSizes();
    SmallVector<OpFoldResult> strides = slice.getMixedStrides();
    if (offsets.size() != rank || sizes.size() != rank || strides.size() != rank)
      return failure();

    Location loc = slice.getLoc();

    // A transpose only renames the axes, so the slice moves with them.
    if (auto transpose = llvm::dyn_cast<linalg::TransposeOp>(producer)) {
      ArrayRef<int64_t> perm = transpose.getPermutation();
      if (perm.size() != rank)
        return failure();
      SmallVector<OpFoldResult> o(rank), s(rank), t(rank);
      SmallVector<int64_t> shape(rank);
      for (unsigned k = 0; k < rank; k++) {
        o[perm[k]] = offsets[k];
        s[perm[k]] = sizes[k];
        t[perm[k]] = strides[k];
        shape[perm[k]] = resTy.getDimSize(k);
      }
      Value inner = rewriter.create<tensor::ExtractSliceOp>(
          loc, RankedTensorType::get(shape, resTy.getElementType()),
          transpose.getInput(), o, s, t);
      Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                    resTy.getElementType());
      rewriter.replaceOpWithNewOp<linalg::TransposeOp>(slice, inner, init, perm);
      return success();
    }

    auto generic = llvm::dyn_cast<linalg::GenericOp>(producer);
    if (!generic || generic.getInputs().size() != 1 ||
        generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity() || !maps[0].isPermutation())
      return failure();
    if (!generic.getRegion().front().getArgument(1).use_empty())
      return failure();
    auto inTy = llvm::dyn_cast<RankedTensorType>(generic.getInputs()[0].getType());
    if (!inTy || inTy.getRank() != rank)
      return failure();

    // The read map says which iteration dimension walks each input dimension,
    // and the iteration is the result, so that is where each input axis is cut.
    SmallVector<OpFoldResult> o(rank), s(rank), t(rank);
    SmallVector<int64_t> shape(rank);
    for (unsigned r = 0; r < rank; r++) {
      auto dim = llvm::dyn_cast<AffineDimExpr>(maps[0].getResult(r));
      if (!dim)
        return failure();
      unsigned k = dim.getPosition();
      o[r] = offsets[k];
      s[r] = sizes[k];
      t[r] = strides[k];
      shape[r] = resTy.getDimSize(k);
    }
    Value inner = rewriter.create<tensor::ExtractSliceOp>(
        loc, RankedTensorType::get(shape, inTy.getElementType()),
        generic.getInputs()[0], o, s, t);
    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                  resTy.getElementType());
    auto moved = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{inner}, ValueRange{init}, maps,
        generic.getIteratorTypesArray());
    rewriter.cloneRegionBefore(generic.getRegion(), moved.getRegion(),
                               moved.getRegion().begin());
    rewriter.replaceOp(slice, moved.getResults());
    return success();
  }
};

/// A map that reads its input broadcast is split in two.
///
/// `--fuse-elementwise-around-matmul` has already put the broadcast and the
/// elementwise work in one region, which reads a small input through a map that
/// drops a dimension and writes the big result: one conversion **per copy**.
/// Doing the work first and broadcasting the answer is strictly less of it --
/// and where the input is a constant it is none at all, because the folder turns
/// the quantized weight into an i8 constant at compile time.
///
/// ConvNeXt is the model this is for. Its MLP weights reach a
/// `linalg.batch_matmul` broadcast into a batch of two, so a 768 x 3072 weight
/// is quantized **twice on every inference**. `--unbatch-single-matmul` already
/// reads through such a broadcast, but only for a batch of one, which this is
/// not. Those weights are 81.4 million elements of ConvNeXt's 83.2 million of
/// scalar work.
///
/// The broadcast itself stays, and that is the right place to stop. Rewriting
/// the `linalg.batch_matmul` to read the small weight directly -- as a
/// `linalg.generic` carrying the contraction's maps with the batch dropped from
/// the weight -- was tried and the model no longer compiles: everything
/// downstream of `--force-quantized-matmul` matches the named operation. What
/// is left after the split is a broadcast of **i8**, which
/// `--gather-to-memref-copy` turns into one `memcpy` per batch of a
/// 2.3 MB run.
class SplitBroadcastOutOfElementwise
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (readsItsIndex(generic))
      return failure();
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity())
      return failure();
    if (!generic.getRegion().front().getArgument(1).use_empty())
      return failure();
    // A body that only copies is the broadcast itself: there is no work to
    // take out of it, and splitting would produce another one to split. That
    // is not a refinement -- the rewrite below writes exactly such a copy, and
    // without this the greedy driver never returns.
    if (isPureGather(generic))
      return failure();

    auto srcTy =
        llvm::dyn_cast<RankedTensorType>(generic.getInputs()[0].getType());
    auto dstTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!srcTy || !dstTy || !srcTy.hasStaticShape() || !dstTy.hasStaticShape())
      return failure();
    // Strictly a broadcast: the read map drops at least one iteration
    // dimension, so the same element is written more than once. An identity
    // read is an ordinary elementwise map and there is nothing to split.
    if ((unsigned)srcTy.getRank() >= maps[0].getNumDims() ||
        srcTy.getNumElements() >= dstTy.getNumElements())
      return failure();
    // Only a plain projection -- each result of the read map is one of the
    // iteration dimensions, in order. That is what torch-mlir writes for a
    // broadcast, and it is what makes the split below exact.
    SmallVector<unsigned> dims;
    for (AffineExpr e : maps[0].getResults()) {
      auto dim = llvm::dyn_cast<AffineDimExpr>(e);
      if (!dim || (!dims.empty() && dim.getPosition() <= dims.back()))
        return failure();
      dims.push_back(dim.getPosition());
    }
    if (dims.size() != (size_t)srcTy.getRank())
      return failure();

    Location loc = generic.getLoc();
    Type elem = dstTy.getElementType();
    MLIRContext *ctx = rewriter.getContext();
    unsigned rank = srcTy.getRank();

    // The work, on the small input.
    SmallVector<AffineMap> smallMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, ctx));
    SmallVector<utils::IteratorType> smallIters(rank,
                                                utils::IteratorType::parallel);
    auto earlyTy = RankedTensorType::get(srcTy.getShape(), elem);
    Value init = rewriter.create<tensor::EmptyOp>(loc, srcTy.getShape(), elem);
    auto early = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{earlyTy}, ValueRange{generic.getInputs()[0]},
        ValueRange{init}, smallMaps, smallIters);
    rewriter.cloneRegionBefore(generic.getRegion(), early.getRegion(),
                               early.getRegion().begin());

    // And the broadcast of the answer, through the same read map.
    Value wideInit =
        rewriter.create<tensor::EmptyOp>(loc, dstTy.getShape(), elem);
    auto spread = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{dstTy.clone(elem)}, ValueRange{early.getResult(0)},
        ValueRange{wideInit}, maps, generic.getIteratorTypesArray(),
        [](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, args[0]);
        });
    rewriter.replaceOp(generic, spread.getResults());
    return success();
  }
};

/// An elementwise map absorbs the relayout it reads.
///
/// The map walks its input with an identity map, so reading a `linalg.transpose`
/// is the same computation as reading the transpose's *source* through the
/// permutation -- and an elementwise body does not care which order the elements
/// arrive in. Composing them means one pass over the data instead of two.
///
/// This is what lets `HoistElementwiseBeforeConcat` below see an Inception
/// block's pooling branch. That branch reads the join, relayouts, pads and
/// pools; `--requantize-before-pooling` and `HoistElementwiseBeforePad` walk its
/// quantization up to just under the relayout and stop there, so the join keeps
/// an f32 reader and the distribution is refused -- and refusing is right,
/// because letting that reader through builds the join **twice**, once in i8
/// and once in f32, which leaves every branch tail with two readers and folding
/// into neither (GoogLeNet lost 12 convolutions and 61.6 million
/// multiply-accumulates to scalar loops that way).
///
/// Composing is only worth it in this direction: the map has to be the
/// transpose's only user, or the relayout is done twice.
class AbsorbTransposeIntoElementwise
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (readsItsIndex(generic))
      return failure();
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    if (!isElementwiseMap(generic))
      return failure();
    auto transpose = generic.getInputs()[0].getDefiningOp<linalg::TransposeOp>();
    if (!transpose || !transpose.getResult()[0].hasOneUse())
      return failure();

    auto srcTy =
        llvm::dyn_cast<RankedTensorType>(transpose.getInput().getType());
    auto resTy =
        llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!srcTy || !resTy || !srcTy.hasStaticShape() || !resTy.hasStaticShape())
      return failure();
    ArrayRef<int64_t> perm = transpose.getPermutation();
    unsigned rank = perm.size();
    if (rank != (unsigned)srcTy.getRank() || rank != (unsigned)resTy.getRank())
      return failure();

    // Read the source through the permutation: result dimension `k` is the
    // source's `perm[k]`, so iteration dimension `k` walks source dimension
    // `perm[k]`, which is the map `(d0, ..) -> (d_{perm^-1[0]}, ..)`.
    SmallVector<AffineExpr> results(rank);
    for (unsigned k = 0; k < rank; k++)
      results[perm[k]] = rewriter.getAffineDimExpr(k);
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineMap> maps{
        AffineMap::get(rank, 0, results, ctx),
        AffineMap::getMultiDimIdentityMap(rank, ctx)};

    Location loc = generic.getLoc();
    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                  resTy.getElementType());
    auto moved = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{transpose.getInput()},
        ValueRange{init}, maps, generic.getIteratorTypesArray());
    rewriter.cloneRegionBefore(generic.getRegion(), moved.getRegion(),
                               moved.getRegion().begin());
    rewriter.replaceOp(generic, moved.getResults());
    return success();
  }
};

/// The same move across a concatenation.
///
/// `torch.cat` is what an Inception block, a DenseNet layer and a detection
/// neck are joined with, and a frontend puts it *before* the requantization: the
/// branches' tails stay f32, so neither convolution ends in a requantization
/// and neither folds. A concatenation only moves elements, so an elementwise
/// operation distributes over it -- the pieces are quantized instead of the
/// join, which is also a quarter of the bytes to copy.
///
/// The pieces all take the consumer's single scale, which is what the
/// calibration measured for the joined activation; that is what a quantized
/// network does with a concatenation anyway, since the result has one scale.
class HoistElementwiseBeforeConcat : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (readsItsIndex(generic))
      return failure();
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity())
      return failure();
    if (!generic.getRegion().front().getArgument(1).use_empty())
      return failure();

    auto concat = generic.getInputs()[0].getDefiningOp<tensor::ConcatOp>();
    if (!concat || concat.getInputs().size() < 2)
      return failure();
    // A join with more than one user, all of them this same operation.
    //
    // An Inception block's branches each quantize the block's *input*, so the
    // join has one identical `linalg.generic` per branch and nothing merges
    // them: a plain `hasOneUse` refuses every one of GoogLeNet's nine joins,
    // and so does every DenseNet layer's. Distributing once and giving each
    // copy the same result is the merge.
    //
    // A global `--cse` is not the way to get it: it merges the contraction
    // tails too, which makes them multi-use, and `matchRequantize` then folds
    // none of them -- GoogLeNet 59 accelerator calls to 49, DenseNet 124 to
    // **6**.
    //
    // Two kinds of user are skipped rather than matched, and the join stays in
    // place for them.
    //
    // A `tensor.pad` is the **pooling branch** of an Inception block: it reads
    // the join and quantizes below the pool. A max-pool commutes with a
    // monotone requantization -- that is what `--requantize-before-pooling`
    // relies on -- so that branch ends up quantized either way.
    //
    // What a user that is *neither* must not be allowed to do is keep the f32
    // join alive. Letting a `linalg.transpose` through was tried and it is much
    // worse than refusing: the join is then built **twice**, once in i8 for the
    // quantized branches and once in f32 for the relayout, so every branch tail
    // has two readers and folds into neither. GoogLeNet lost 12 convolutions to
    // scalar loops that way -- **61.6 million** multiply-accumulates -- against
    // 585 thousand elements saved elsewhere.
    //
    // All of it is only sound because the join's quantization is **one** scale,
    // the one the calibration measured for the joined activation; every branch
    // takes that same scale, which is what a quantized network does with a
    // concatenation anyway.
    SmallVector<linalg::GenericOp> sameUsers;
    for (Operation *u : concat->getUsers()) {
      if (llvm::isa<tensor::PadOp>(u))
        continue;
      auto other = llvm::dyn_cast<linalg::GenericOp>(u);
      // Not `OperationEquivalence::isEquivalentTo`: each copy has its own
      // `tensor.empty` for the destination, so comparing operands says no to
      // two operations that compute exactly the same thing. What has to match
      // is the input, the maps, the iterators, the result type and the body.
      if (!other || other.getInputs().size() != 1 ||
          other.getOutputs().size() != 1 ||
          other.getInputs()[0] != generic.getInputs()[0] ||
          other.getIndexingMapsArray() != maps ||
          other.getIteratorTypesArray() != generic.getIteratorTypesArray() ||
          other.getResult(0).getType() != generic.getResult(0).getType() ||
          !OperationEquivalence::isRegionEquivalentTo(
              &other.getRegion(), &generic.getRegion(),
              OperationEquivalence::IgnoreLocations))
        return failure();
      sameUsers.push_back(other);
    }
    if (sameUsers.empty())
      return failure();
    auto srcTy = llvm::dyn_cast<RankedTensorType>(concat.getType());
    auto resTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!srcTy || !resTy || !srcTy.hasStaticShape() || !resTy.hasStaticShape())
      return failure();

    // The read map says which iteration dimension walks each source dimension,
    // which is also where the joined axis ends up.
    unsigned rank = maps[0].getNumResults();
    SmallVector<int64_t> order;
    if (rank != (unsigned)resTy.getRank() ||
        !readOrder(maps[0], srcTy.getShape(), order))
      return failure();

    // Build at the concatenation, not at this operation. The result now feeds
    // several users and one of them can sit above this one; without the move
    // the rewrite produces a use that its definition does not dominate.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(concat);

    Location loc = generic.getLoc();
    Type elem = resTy.getElementType();
    SmallVector<Value> pieces;
    for (Value piece : concat.getInputs()) {
      auto pieceTy = llvm::dyn_cast<RankedTensorType>(piece.getType());
      if (!pieceTy || !pieceTy.hasStaticShape() || pieceTy.getRank() != rank)
        return failure();
      SmallVector<int64_t> shape(rank, 0);
      for (unsigned r = 0; r < rank; r++)
        shape[order[r]] = pieceTy.getShape()[r];
      Value init = rewriter.create<tensor::EmptyOp>(loc, shape, elem);
      auto one = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{RankedTensorType::get(shape, elem)}, ValueRange{piece},
          ValueRange{init}, generic.getIndexingMapsArray(),
          generic.getIteratorTypesArray());
      rewriter.cloneRegionBefore(generic.getRegion(), one.getRegion(),
                                 one.getRegion().begin());
      pieces.push_back(one.getResult(0));
    }

    Value joined = rewriter.create<tensor::ConcatOp>(
        loc, resTy, order[concat.getDim()], pieces);
    for (linalg::GenericOp user : sameUsers)
      rewriter.replaceOp(user, joined);
    return success();
  }
};

/// The same move across the zero-stuffing a transposed convolution arrives as.
///
/// torch-mlir writes `ConvTranspose2d` as an ordinary convolution over an input
/// with `stride - 1` zeros inserted between its samples: a zero-filled buffer
/// and a strided `tensor.insert_slice` into it. Quantizing after that converts
/// the whole stuffed buffer -- four times the elements for a stride of two, and
/// three quarters of them zeros. Quantizing first converts only the real ones,
/// and the stuffing then moves i8.
///
/// Sound for the same reason the padding case is: the operation being moved
/// takes zero to zero, so the zeros it would have produced are the zeros the
/// fill already put there.
class HoistElementwiseBeforeStuffing
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (readsItsIndex(generic))
      return failure();
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity() || !maps[0].isPermutation())
      return failure();
    if (!generic.getRegion().front().getArgument(1).use_empty())
      return failure();

    auto insert = generic.getInputs()[0].getDefiningOp<tensor::InsertSliceOp>();
    if (!insert || !insert->hasOneUse())
      return failure();
    auto fill = insert.getDest().getDefiningOp<linalg::FillOp>();
    if (!fill || fill.getInputs().size() != 1 ||
        !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
      return failure();

    auto srcTy = llvm::dyn_cast<RankedTensorType>(insert.getSource().getType());
    auto resTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!srcTy || !resTy || !srcTy.hasStaticShape() || !resTy.hasStaticShape())
      return failure();

    Block &body = generic.getRegion().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        !mapsZeroToZero(yield.getOperand(0), body.getArgument(0)))
      return failure();

    unsigned rank = maps[0].getNumResults();
    if (rank != srcTy.getRank() || rank != resTy.getRank())
      return failure();
    SmallVector<int64_t> order(rank, 0);
    for (unsigned r = 0; r < rank; r++) {
      auto dim = llvm::dyn_cast<AffineDimExpr>(maps[0].getResult(r));
      if (!dim)
        return failure();
      order[r] = dim.getPosition();
    }

    Location loc = generic.getLoc();
    Type elem = resTy.getElementType();
    SmallVector<int64_t> earlyShape(rank, 0);
    for (unsigned r = 0; r < rank; r++)
      earlyShape[order[r]] = srcTy.getShape()[r];
    auto earlyTy = RankedTensorType::get(earlyShape, elem);
    Value init = rewriter.create<tensor::EmptyOp>(loc, earlyTy.getShape(), elem);
    auto early = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{earlyTy}, ValueRange{insert.getSource()}, ValueRange{init},
        generic.getIndexingMapsArray(), generic.getIteratorTypesArray());
    rewriter.cloneRegionBefore(generic.getRegion(), early.getRegion(),
                               early.getRegion().begin());

    SmallVector<OpFoldResult> offsetsIn = insert.getMixedOffsets();
    SmallVector<OpFoldResult> sizesIn = insert.getMixedSizes();
    SmallVector<OpFoldResult> stridesIn = insert.getMixedStrides();
    SmallVector<OpFoldResult> offsets(rank), sizes(rank), strides(rank);
    for (unsigned r = 0; r < rank; r++) {
      offsets[order[r]] = offsetsIn[r];
      sizes[order[r]] = sizesIn[r];
      strides[order[r]] = stridesIn[r];
    }

    Value zero = rewriter.create<arith::ConstantOp>(loc, elem,
                                                    rewriter.getZeroAttr(elem));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(), elem);
    Value zeroed = rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        generic, early.getResult(0), zeroed, offsets, sizes, strides);
    return success();
  }
};

/// `elementwise(gather(x))` computes the same values as `gather(elementwise(x))`
/// -- a gather only moves elements around -- but the second form does the work
/// on the *unexpanded* operand and moves the smaller result.
///
/// im2col expands its input by the kernel footprint, so quantizing after it
/// converts nine times as many elements as quantizing before, and the gather
/// then copies i8 rather than f32.
class HoistElementwise : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp elementwise,
                                PatternRewriter &rewriter) const final {
    if (readsItsIndex(elementwise))
      return failure();
    if (!isElementwiseMap(elementwise))
      return failure();

    // The packing's result is usually reshaped before it is used, so look
    // through a single collapse or expand on the way to it. The reshape is
    // rebuilt on the converted values afterwards.
    Value source = elementwise.getInputs()[0];
    Operation *reshape = nullptr;
    if (auto collapse = source.getDefiningOp<tensor::CollapseShapeOp>()) {
      reshape = collapse;
      source = collapse.getSrc();
    } else if (auto expand = source.getDefiningOp<tensor::ExpandShapeOp>()) {
      reshape = expand;
      source = expand.getSrc();
    }
    if (reshape && !reshape->getResult(0).hasOneUse())
      return failure();

    auto gather = source.getDefiningOp<linalg::GenericOp>();
    if (!gather || !isPureGather(gather))
      return failure();
    // Moving it only pays if nothing else still needs the gathered values.
    if (!gather.getResult(0).hasOneUse())
      return failure();

    auto srcTy = llvm::dyn_cast<RankedTensorType>(gather.getInputs()[0].getType());
    auto midTy = llvm::dyn_cast<RankedTensorType>(gather.getResult(0).getType());
    auto gatheredTy = midTy;
    auto dstTy = llvm::dyn_cast<RankedTensorType>(elementwise.getResult(0).getType());
    if (!srcTy || !midTy || !dstTy || !srcTy.hasStaticShape() ||
        !dstTy.hasStaticShape())
      return failure();
    if (midTy.getNumElements() != dstTy.getNumElements())
      return failure();

    Location loc = elementwise.getLoc();
    Type newElem = dstTy.getElementType();

    // The elementwise operation, now over the gather's input.
    auto smallTy = RankedTensorType::get(srcTy.getShape(), newElem);
    Value smallInit = rewriter.create<tensor::EmptyOp>(loc, smallTy, ValueRange{});
    SmallVector<AffineMap> identity(
        2, rewriter.getMultiDimIdentityMap(srcTy.getRank()));
    SmallVector<utils::IteratorType> parallel(srcTy.getRank(),
                                              utils::IteratorType::parallel);
    auto moved = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{smallTy}, ValueRange{gather.getInputs()[0]},
        ValueRange{smallInit}, identity, parallel);
    rewriter.cloneRegionBefore(elementwise.getRegion(), moved.getRegion(),
                               moved.getRegion().end());

    // The gather, now over the converted values, keeping its own shape.
    auto newGatheredTy =
        RankedTensorType::get(gatheredTy.getShape(), newElem);
    Value gatherInit =
        rewriter.create<tensor::EmptyOp>(loc, newGatheredTy, ValueRange{});
    auto newGather = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{newGatheredTy}, ValueRange{moved.getResult(0)},
        ValueRange{gatherInit}, gather.getIndexingMapsArray(),
        gather.getIteratorTypesArray());
    rewriter.cloneRegionBefore(gather.getRegion(), newGather.getRegion(),
                               newGather.getRegion().end());
    // The cloned body still names the old element type.
    Block &gatherBody = newGather.getRegion().front();
    gatherBody.getArgument(0).setType(newElem);
    gatherBody.getArgument(1).setType(newElem);

    // Put the reshape back, now on i8.
    Value result = newGather.getResult(0);
    if (auto collapse = llvm::dyn_cast_or_null<tensor::CollapseShapeOp>(reshape))
      result = rewriter.create<tensor::CollapseShapeOp>(
          loc, dstTy, result, collapse.getReassociationIndices());
    else if (auto expand = llvm::dyn_cast_or_null<tensor::ExpandShapeOp>(reshape))
      result = rewriter.create<tensor::ExpandShapeOp>(
          loc, dstTy, result, expand.getReassociationIndices());

    rewriter.replaceOp(elementwise, result);
    return success();
  }
};

/// `transpose(elementwise(x))` written as one operation, taken apart again.
///
/// A transformer's `K.T` reaches the quantizer as a transposing copy sitting
/// between the projection's dequantization and the requantization that feeds
/// `Q @ K.T`. Fusion collapses the three into a single pass that reads the
/// accumulator permuted, and the permutation is what makes it unfoldable: the
/// accelerator writes its result in row order, so `matchRequantize` needs the
/// requantization to write in that order too. The whole tail then stays a
/// scalar f32 pass over the accumulator.
///
/// Split apart, the requantization is back against the matmul -- where it
/// folds into `matmul_i8_scale` -- and what is left is a copy that moves i8
/// instead of i32. Only worth doing in that direction, which is what the width
/// test says.
class SplitTransposeOutOfElementwise
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (readsItsIndex(generic))
      return failure();
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity() || !maps[0].isPermutation() ||
        maps[0].isIdentity())
      return failure();
    Block &body = generic.getRegion().front();
    if (!body.getArgument(1).use_empty())
      return failure();

    // Only on a contraction's accumulator. Everywhere else the permutation is
    // a layout rewrite that has nothing to fold into, and taking it out of the
    // quantization is a plain loss: it put a separate copy between a
    // convolution and its input, which is exactly what stops the convolution
    // folding. Measured over the model set, splitting unconditionally took
    // ResNet-20 from 21 offloaded convolutions to 12 and MobileNetV2 from 35 to
    // 18.
    auto producer =
        generic.getInputs()[0].getDefiningOp<linalg::LinalgOp>();
    if (!producer || !linalg::isaContractionOpInterface(producer) ||
        !producer->getResult(0).hasOneUse())
      return failure();

    auto inTy = llvm::dyn_cast<RankedTensorType>(generic.getInputs()[0].getType());
    auto outTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return failure();
    Type inElem = inTy.getElementType();
    Type outElem = outTy.getElementType();
    if (!inElem.isIntOrFloat() || !outElem.isIntOrFloat())
      return failure();
    // A copy is a copy whichever side the conversion happens on; splitting
    // only pays when it leaves the narrower type to move.
    if (outElem.getIntOrFloatBitWidth() >= inElem.getIntOrFloatBitWidth())
      return failure();

    Location loc = generic.getLoc();
    auto narrowTy = RankedTensorType::get(inTy.getShape(), outElem);
    Value init = rewriter.create<tensor::EmptyOp>(loc, narrowTy, ValueRange{});
    SmallVector<AffineMap> identity(
        2, rewriter.getMultiDimIdentityMap(inTy.getRank()));
    auto converted = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{narrowTy}, ValueRange{generic.getInputs()[0]},
        ValueRange{init}, identity, generic.getIteratorTypesArray());
    rewriter.cloneRegionBefore(generic.getRegion(), converted.getRegion(),
                               converted.getRegion().end());

    // The copy that is left keeps the original maps, so it moves exactly the
    // elements the fused operation did -- only now they are i8.
    Value copyInit =
        rewriter.create<tensor::EmptyOp>(loc, outTy, ValueRange{});
    auto copy = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{outTy}, ValueRange{converted.getResult(0)},
        ValueRange{copyInit}, maps, generic.getIteratorTypesArray(),
        [](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, args[0]);
        });
    rewriter.replaceOp(generic, copy.getResults());
    return success();
  }
};

class HoistElementwiseBeforeGather
    : public impl::HoistElementwiseBeforeGatherBase<HoistElementwiseBeforeGather> {
public:
  using impl::HoistElementwiseBeforeGatherBase<
      HoistElementwiseBeforeGather>::HoistElementwiseBeforeGatherBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<HoistElementwise, HoistElementwiseBeforePad,
                 SplitBroadcastOutOfElementwise,
                 MovePadThroughTranspose, ZeroPadAMaxPool,
                 HoistElementwiseBeforeConcat, AbsorbTransposeIntoElementwise,
                 MoveSliceBeforeElementwise,
                 HoistElementwiseBeforeStuffing, MoveSliceBeforePad,
                 SplitTransposeOutOfElementwise>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
