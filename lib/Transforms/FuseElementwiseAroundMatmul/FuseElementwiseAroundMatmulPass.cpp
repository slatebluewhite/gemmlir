//===- FuseElementwiseAroundMatmulPass.cpp ------------------*- C++ -*-===//
//
// Fuses elementwise work without fusing anything into a matmul, and without
// fusing into a consumer that would run it more times than it has elements.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"
#include "Gemmlir/GemmlirPatterns.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FUSEELEMENTWISEAROUNDMATMUL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// `collapse_shape(broadcast(x))` is a broadcast into the collapsed shape.
///
/// A frontend materialises a bias by broadcasting it over the whole activation
/// and then reshaping that to feed the layer: torch-mlir turns 8 floats into a
/// 1x8x16x16 tensor and collapses it to 8x256. Fusion would absorb the
/// broadcast into its consumer -- the iteration spaces match -- but the reshape
/// sits in between and neither of MLIR's reshape-fusion directions moves it
/// (by-expansion widens the consumer back to 4-D and loses the matmul match;
/// by-collapsing only handles `expand_shape` with its producer). Sinking the
/// reshape into the broadcast leaves the two adjacent, and ordinary fusion then
/// removes the materialisation entirely: 2832 elements on the two-layer CNN.
///
/// Sound when every reassociation group either is not read at all or contains
/// exactly one read dimension with every other dimension in it of extent 1 --
/// then the collapsed index *is* that dimension's index. Anything else would
/// need floordiv/mod to undo and is left alone.
class SinkCollapseIntoBroadcast
    : public OpRewritePattern<tensor::CollapseShapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapse,
                                PatternRewriter &rewriter) const final {
    auto generic = collapse.getSrc().getDefiningOp<linalg::GenericOp>();
    if (!generic || !generic->hasOneUse())
      return failure();
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();

    Block &body = generic.getRegion().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != body.getArgument(0))
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(), linalg::isParallelIterator))
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity() ||
        !maps[0].isProjectedPermutation())
      return failure();

    auto srcTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    auto resTy = collapse.getResultType();
    if (!srcTy || !srcTy.hasStaticShape() || !resTy.hasStaticShape())
      return failure();
    ArrayRef<int64_t> srcShape = srcTy.getShape();

    SmallVector<ReassociationIndices> groups = collapse.getReassociationIndices();
    SmallVector<int64_t> dimToGroup(srcShape.size(), -1);
    for (auto [g, idxs] : llvm::enumerate(groups))
      for (int64_t d : idxs)
        dimToGroup[d] = g;

    llvm::SmallDenseSet<int64_t> read;
    for (AffineExpr e : maps[0].getResults())
      read.insert(llvm::cast<AffineDimExpr>(e).getPosition());

    for (ReassociationIndices &idxs : groups) {
      unsigned reads = 0;
      for (int64_t d : idxs)
        reads += read.contains(d);
      if (reads == 0)
        continue;
      if (reads > 1)
        return failure();
      for (int64_t d : idxs)
        if (!read.contains(d) && srcShape[d] != 1)
          return failure();
    }

    Location loc = generic.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> reads;
    for (AffineExpr e : maps[0].getResults())
      reads.push_back(rewriter.getAffineDimExpr(
          dimToGroup[llvm::cast<AffineDimExpr>(e).getPosition()]));

    unsigned rank = groups.size();
    SmallVector<AffineMap> newMaps = {
        AffineMap::get(rank, 0, reads, ctx),
        AffineMap::getMultiDimIdentityMap(rank, ctx)};
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                  resTy.getElementType());
    auto broadcast = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{generic.getInputs()[0]},
        ValueRange{init}, newMaps, iters,
        [](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, args[0]);
        });
    rewriter.replaceOp(collapse, broadcast.getResult(0));
    return success();
  }
};

/// An elementwise operation on `expand_shape(x)` is the same operation on `x`.
///
/// The other half of the reshape problem. torch-mlir keeps reshaping between
/// the 2-D form a contraction wants and the 4-D form an activation has, so the
/// dequantize lands on 8x256 and the relu-and-requantize that follows it on
/// 1x8x16x16, with a view in between. They have the same iteration space and
/// would fuse into one loop over one buffer, except for the reshape -- and
/// neither of MLIR's directions folds an `expand_shape` with its *consumer* by
/// collapsing that consumer.
///
/// Collapsing it is the right direction here: it keeps the work 2-D, next to
/// the matmul, where --convert-linalg-to-gemmlir can still see everything.
/// Expanding the producer instead widens operations back to 4-D and loses the
/// matmul match entirely.
///
/// Other operands come along when their own map survives the collapse, which a
/// per-channel bias does: it reads the last dimension, and that dimension is a
/// reassociation group of its own.
class CollapseElementwiseOverExpand : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getOutputs().size() != 1 || generic.getInputs().empty())
      return failure();
    if (!generic.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(), linalg::isParallelIterator))
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != generic.getInputs().size() + 1 || !maps.back().isIdentity())
      return failure();

    auto resTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!resTy || !resTy.hasStaticShape())
      return failure();
    ArrayRef<int64_t> shape = resTy.getShape();

    // An input that is a whole-shape view fixes the reassociation to undo.
    tensor::ExpandShapeOp anchor;
    for (auto [in, m] : llvm::zip(generic.getInputs(), maps)) {
      auto e = in.getDefiningOp<tensor::ExpandShapeOp>();
      if (!e || !isViewOfIterationSpace(m, shape))
        continue;
      auto ty = llvm::cast<RankedTensorType>(e.getType());
      if (ty.getShape() != shape || !ty.hasStaticShape())
        continue;
      anchor = e;
      break;
    }
    if (!anchor)
      return failure();

    SmallVector<ReassociationIndices> groups = anchor.getReassociationIndices();
    SmallVector<int64_t> dimToGroup(shape.size(), -1);
    for (auto [g, idxs] : llvm::enumerate(groups))
      for (int64_t d : idxs)
        dimToGroup[d] = g;

    // Decide what each input becomes on the collapsed iteration space: an
    // `expand_shape` of the same shape gives up its source, anything else keeps
    // its operand and has its map rewritten -- which needs each group to hold at
    // most one dimension it reads, the rest being of extent 1.
    unsigned rank = groups.size();
    MLIRContext *ctx = rewriter.getContext();
    AffineMap identity = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<Value> newInputs;
    SmallVector<AffineMap> newMaps;
    for (auto [in, m] : llvm::zip(generic.getInputs(), maps)) {
      if (isViewOfIterationSpace(m, shape)) {
        auto e = in.getDefiningOp<tensor::ExpandShapeOp>();
        if (e && llvm::cast<RankedTensorType>(e.getType()).getShape() == shape &&
            e.getReassociationIndices() == groups) {
          newInputs.push_back(e.getSrc());
          newMaps.push_back(identity);
          continue;
        }
        return failure();
      }
      if (!m.isProjectedPermutation())
        return failure();
      llvm::SmallDenseSet<int64_t> readDims;
      for (AffineExpr e : m.getResults())
        readDims.insert(llvm::cast<AffineDimExpr>(e).getPosition());
      for (ReassociationIndices &idxs : groups) {
        unsigned reads = 0;
        for (int64_t d : idxs)
          reads += readDims.contains(d);
        if (reads > 1)
          return failure();
        if (reads == 1)
          for (int64_t d : idxs)
            if (!readDims.contains(d) && shape[d] != 1)
              return failure();
      }
      SmallVector<AffineExpr> results;
      for (AffineExpr e : m.getResults())
        results.push_back(rewriter.getAffineDimExpr(
            dimToGroup[llvm::cast<AffineDimExpr>(e).getPosition()]));
      newInputs.push_back(in);
      newMaps.push_back(AffineMap::get(rank, 0, results, ctx));
    }
    newMaps.push_back(identity);

    auto srcTy = llvm::cast<RankedTensorType>(anchor.getSrc().getType());
    Location loc = generic.getLoc();
    auto collapsedResTy =
        RankedTensorType::get(srcTy.getShape(), resTy.getElementType());
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    Value init = rewriter.create<tensor::EmptyOp>(loc, srcTy.getShape(),
                                                  resTy.getElementType());
    auto collapsed = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{collapsedResTy}, newInputs, ValueRange{init}, newMaps,
        iters);
    rewriter.cloneRegionBefore(generic.getRegion(), collapsed.getRegion(),
                               collapsed.getRegion().begin());

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        generic, resTy, collapsed.getResult(0), groups);
    return success();
  }

private:
  /// True when `m` reads one element per iteration of the full space: the
  /// identity, except that a frontend writes a constant 0 rather than the
  /// dimension wherever an axis has extent 1.
  static bool isViewOfIterationSpace(AffineMap m, ArrayRef<int64_t> shape) {
    if (m.getNumResults() != shape.size())
      return false;
    for (auto [r, e] : llvm::enumerate(m.getResults())) {
      if (auto dim = llvm::dyn_cast<AffineDimExpr>(e)) {
        if (dim.getPosition() != r)
          return false;
        continue;
      }
      auto cst = llvm::dyn_cast<AffineConstantExpr>(e);
      if (!cst || cst.getValue() != 0 || shape[r] != 1)
        return false;
    }
    return true;
  }
};

/// The one value an operand carries everywhere belongs in the body.
///
/// torch-mlir lowers a bounded activation -- `ReLU6`, `Hardtanh` -- by putting
/// each bound in a 0-D tensor and broadcasting it, so what reaches the
/// requantization is not `max(x, 0)` against a constant but against a block
/// argument. Nothing that reads a body can see through that: the relu matcher
/// looks for a zero and finds an argument, and a whole MobileNet block's
/// convolutions stay scalar loops because of it.
class FoldConstantOperand : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  /// The single value an operand is made of, if it is made of one.
  static TypedAttr uniformValue(Value v) {
    // A slice or a reshape of something uniform is uniform. A grouped
    // convolution's accumulator is one zero fill sliced per group, and the
    // slice is what reaches the tail: `gmin` was writing 8192 f32 zeros and
    // reading every one of them back to add nothing.
    for (unsigned step = 0; step < 8; step++) {
      if (auto slice = v.getDefiningOp<tensor::ExtractSliceOp>()) {
        v = slice.getSource();
        continue;
      }
      if (auto expand = v.getDefiningOp<tensor::ExpandShapeOp>()) {
        v = expand.getSrc();
        continue;
      }
      if (auto collapse = v.getDefiningOp<tensor::CollapseShapeOp>()) {
        v = collapse.getSrc();
        continue;
      }
      break;
    }
    if (auto fill = v.getDefiningOp<linalg::FillOp>()) {
      if (fill.getInputs().size() != 1)
        return {};
      Attribute a;
      if (matchPattern(fill.getInputs()[0], m_Constant(&a)))
        return llvm::dyn_cast<TypedAttr>(a);
      return {};
    }
    if (auto producer = v.getDefiningOp<linalg::GenericOp>()) {
      if (!producer.getInputs().empty() || producer.getOutputs().size() != 1)
        return {};
      auto yield = llvm::dyn_cast<linalg::YieldOp>(
          producer.getRegion().front().getTerminator());
      if (!yield || yield.getNumOperands() != 1)
        return {};
      Attribute a;
      if (matchPattern(yield.getOperand(0), m_Constant(&a)))
        return llvm::dyn_cast<TypedAttr>(a);
      return {};
    }
    DenseElementsAttr dense;
    if (matchPattern(v, m_Constant(&dense)) && dense.isSplat())
      return llvm::dyn_cast<TypedAttr>(dense.getSplatValue<Attribute>());
    return {};
  }

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getOutputs().size() != 1 || generic->getNumResults() != 1)
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != generic.getInputs().size() + 1)
      return failure();

    unsigned k = 0;
    TypedAttr value;
    for (; k < generic.getInputs().size(); k++)
      if ((value = uniformValue(generic.getInputs()[k])))
        break;
    if (!value)
      return failure();

    SmallVector<Value> ins;
    SmallVector<AffineMap> newMaps;
    for (unsigned i = 0; i < generic.getInputs().size(); i++) {
      if (i == k)
        continue;
      ins.push_back(generic.getInputs()[i]);
      newMaps.push_back(maps[i]);
    }
    newMaps.push_back(maps.back());

    Block &old = generic.getRegion().front();
    auto replacement = rewriter.create<linalg::GenericOp>(
        generic.getLoc(), generic->getResultTypes(), ins, generic.getOutputs(),
        newMaps, generic.getIteratorTypesArray(),
        [&](OpBuilder &b, Location l, ValueRange args) {
          IRMapping map;
          unsigned next = 0;
          for (unsigned i = 0; i < old.getNumArguments(); i++) {
            if (i == k) {
              map.map(old.getArgument(i), b.create<arith::ConstantOp>(l, value));
              continue;
            }
            map.map(old.getArgument(i), args[next++]);
          }
          for (Operation &op : old.without_terminator())
            b.clone(op, map);
          auto yield = llvm::cast<linalg::YieldOp>(old.getTerminator());
          b.create<linalg::YieldOp>(l, map.lookup(yield.getOperand(0)));
        });
    rewriter.replaceOp(generic, replacement.getResults());
    return success();
  }
};

/// A requantization whose scales cancel, over a value that is already i8, is a
/// copy.
///
/// `--share-branch-quantization` quantizes a branching activation once and
/// hands the other consumers the dequantization of it. Where that other
/// consumer is a second convolution reading the *same* tensor -- a ResNet stage
/// transition, whose 1x1 projection and 3x3 convolution both read the block's
/// input -- the calibration measured the same range for both, so it quantizes
/// again at the scale it was just dequantized at. What is left is
/// `clip(round(q * s / s))`, a full pass over the activation that computes `q`.
///
/// This is the fold the quant dialect does for `qcast(dcast(x))` and that the
/// sharing has to write out as arithmetic to avoid (it would otherwise undo
/// itself); doing it here, once the scales are constants in a body, is the
/// honest version of it. Exact: `q` is a small integer, `(q*s)/s` is within
/// 1.5e-5 of it for `|q| <= 127`, and `roundeven` recovers it.
/// Walks a quantization's body from its `linalg.yield` back through the
/// truncation, the clamp, the rounding and the scaling, and reports the ratio
/// it multiplies by and the value the chain ends at.
///
/// The two sides of the ratio are kept apart rather than divided as they are
/// found: the chain is walked from the bottom, so a scale of `c` over `c` would
/// be computed as `(1/c)*c`, which for most `c` is not 1.
static bool matchQuantizeChain(linalg::GenericOp generic, double *numerator,
                               double *denominator, Value *end) {
  Block &body = generic.getRegion().front();
  if (!body.getArguments().back().use_empty())
    return false;
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto trunc = yield.getOperand(0).getDefiningOp<arith::TruncIOp>();
  if (!trunc || !trunc.getType().isInteger(8))
    return false;

  // The clamp has to be inert: anything narrower than i8 is a real operation.
  Value clamped = trunc.getIn();
  int64_t lo = 0, hi = 0;
  bool sawLo = false, sawHi = false;
  while (true) {
    APInt c;
    if (auto min = clamped.getDefiningOp<arith::MinSIOp>()) {
      if (!matchPattern(min.getRhs(), m_ConstantInt(&c)))
        return false;
      hi = c.getSExtValue();
      sawHi = true;
      clamped = min.getLhs();
      continue;
    }
    if (auto max = clamped.getDefiningOp<arith::MaxSIOp>()) {
      if (!matchPattern(max.getRhs(), m_ConstantInt(&c)))
        return false;
      lo = c.getSExtValue();
      sawLo = true;
      clamped = max.getLhs();
      continue;
    }
    break;
  }
  if (!sawLo || !sawHi || lo > -128 || hi < 127)
    return false;

  auto toInt = clamped.getDefiningOp<arith::FPToSIOp>();
  if (!toInt)
    return false;
  auto round = toInt.getIn().getDefiningOp<math::RoundEvenOp>();
  if (!round)
    return false;

  auto constant = [](Value c, double *out) {
    llvm::APFloat f(0.0f);
    if (!matchPattern(c, m_ConstantFloat(&f)))
      return false;
    *out = f.convertToDouble();
    return true;
  };
  *numerator = 1.0;
  *denominator = 1.0;
  Value v = round.getOperand();
  while (true) {
    double c = 0.0;
    if (auto mul = v.getDefiningOp<arith::MulFOp>()) {
      if (constant(mul.getRhs(), &c))
        v = mul.getLhs();
      else if (constant(mul.getLhs(), &c))
        v = mul.getRhs();
      else
        return false;
      if (!(c > 0.0))
        return false;
      *numerator *= c;
      continue;
    }
    if (auto div = v.getDefiningOp<arith::DivFOp>()) {
      if (!constant(div.getRhs(), &c) || !(c > 0.0))
        return false;
      *denominator *= c;
      v = div.getLhs();
      continue;
    }
    break;
  }
  *end = v;
  return true;
}

/// The scale an `i8 -> f32` dequantization multiplies by, if that is all it is.
static std::optional<double> dequantizeScaleOf(linalg::GenericOp generic) {
  if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
    return std::nullopt;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return std::nullopt;
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity())
    return std::nullopt;
  if (!getElementTypeOrSelf(generic.getInputs()[0].getType()).isInteger(8))
    return std::nullopt;
  Block &body = generic.getRegion().front();
  if (!body.getArgument(1).use_empty())
    return std::nullopt;
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  auto mul = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
  if (!mul)
    return std::nullopt;
  llvm::APFloat f(0.0f);
  Value widened;
  if (matchPattern(mul.getRhs(), m_ConstantFloat(&f)))
    widened = mul.getLhs();
  else if (matchPattern(mul.getLhs(), m_ConstantFloat(&f)))
    widened = mul.getRhs();
  else
    return std::nullopt;
  double scale = f.convertToDouble();
  if (!(scale > 0.0))
    return std::nullopt;
  auto toFloat = widened.getDefiningOp<arith::SIToFPOp>();
  if (!toFloat || toFloat.getIn() != body.getArgument(0))
    return std::nullopt;
  return scale;
}

/// `quantize_s(dequantize_s(x))` is `x`.
///
/// `--share-branch-quantization` gives the branches of a split one scale, so an
/// activation that feeds two of them is dequantized to f32 once and quantized
/// straight back -- at the same scale -- once per branch. Fusion will not merge
/// the two halves because the dequantization has more than one consumer, and
/// recomputing it per branch is what that rule exists to prevent; but here the
/// pair *disappears*, so there is nothing to recompute. On `atr` it is three
/// passes of 9216 elements, a third of everything the model has left in
/// software.
///
/// Exact, not approximate: `x * s` is a float with the same significand as `x`
/// for any `x` an i8 can hold, so `(x * s) / s` is within a relative 1e-7 of
/// `x` and `roundeven` returns it. The clamp cannot fire on a value that came
/// out of an i8.
class DequantizeThenQuantizeIsACopy : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity())
      return failure();

    // The pair may have a slice between it: a grouped convolution dequantizes
    // the joined activation once and takes one slice per group, and each of
    // those is quantized back at the same scale. Cutting the i8 instead is the
    // same values, and it leaves everything above the dequantization alone --
    // which matters, because restructuring that far up costs a convolution its
    // fold (`grp` 19.2 -> 22.6 ms when the slices were moved instead).
    Value quantizeInput = generic.getInputs()[0];
    auto slice = quantizeInput.getDefiningOp<tensor::ExtractSliceOp>();
    if (slice)
      quantizeInput = slice.getSource();

    auto dequantize = quantizeInput.getDefiningOp<linalg::GenericOp>();
    if (!dequantize)
      return failure();
    std::optional<double> scale = dequantizeScaleOf(dequantize);
    if (!scale)
      return failure();

    Value source = dequantize.getInputs()[0];
    if (!slice && source.getType() != generic->getResult(0).getType())
      return failure();
    if (slice) {
      auto wide = llvm::dyn_cast<RankedTensorType>(slice.getType());
      auto narrow =
          llvm::dyn_cast<RankedTensorType>(generic->getResult(0).getType());
      auto srcTy = llvm::dyn_cast<RankedTensorType>(source.getType());
      if (!wide || !narrow || !srcTy || wide.getShape() != narrow.getShape())
        return failure();
    }

    double numerator = 1.0, denominator = 1.0;
    Value end;
    if (!matchQuantizeChain(generic, &numerator, &denominator, &end))
      return failure();
    if (end != generic.getRegion().front().getArgument(0))
      return failure();
    // The quantization divides by what the dequantization multiplied by.
    if (numerator * *scale != denominator)
      return failure();

    if (slice) {
      auto narrow =
          llvm::cast<RankedTensorType>(generic->getResult(0).getType());
      rewriter.setInsertionPoint(generic);
      source = rewriter.create<tensor::ExtractSliceOp>(
          generic.getLoc(),
          RankedTensorType::get(
              narrow.getShape(),
              getElementTypeOrSelf(dequantize.getInputs()[0].getType())),
          source, slice.getMixedOffsets(), slice.getMixedSizes(),
          slice.getMixedStrides());
    }
    rewriter.replaceOp(generic, source);
    return success();
  }
};

/// `transpose(f(x))` is one pass, not two.
///
/// A model that ends in NCHW dequantizes its last accumulator in NHWC and then
/// relayouts the result, and the relayout is a `linalg.transpose` -- a named
/// operation, which the elementwise fusion does not fuse into. Two full passes
/// over the activation where one would do: on `atr` and `atrn` that is 9216
/// elements of f32 written and read back for nothing.
///
/// The fused form iterates the *producer's* space and writes permuted, so the
/// reads stay in order and only the writes scatter -- which is the way round
/// that costs less.
class SinkElementwiseIntoTranspose
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    if (!transpose.hasPureTensorSemantics())
      return failure();
    // Only where the relayout is the last thing the function does. Fusing one
    // in the middle takes the elementwise operation out of reach of everything
    // that would otherwise have fused *it*, and the model set says that costs
    // more than the pass it saves: `atr` 63340 scalar elements to 81772,
    // `shf` and `shu` 2826 to 6922.
    if (!llvm::all_of(transpose.getResult()[0].getUsers(), [](Operation *user) {
          return user->hasTrait<OpTrait::ReturnLike>();
        }))
      return failure();

    auto producer = transpose.getInput().getDefiningOp<linalg::GenericOp>();
    if (!producer || !producer->hasOneUse() || producer.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(producer.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    SmallVector<AffineMap> maps = producer.getIndexingMapsArray();
    if (maps.empty() || !maps.back().isIdentity())
      return failure();
    if (!producer.getRegion().front().getArguments().back().use_empty())
      return failure();

    auto inTy = llvm::dyn_cast<RankedTensorType>(producer.getResult(0).getType());
    auto outTy = llvm::dyn_cast<RankedTensorType>(transpose.getResult()[0].getType());
    if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
      return failure();
    ArrayRef<int64_t> perm = transpose.getPermutation();
    if (perm.size() != (size_t)inTy.getRank())
      return failure();

    // `linalg.transpose` gives `dim(result, k) == dim(input, perm[k])`, so
    // iterating the input's index `j` writes `result[i]` with
    // `i[k] = j[perm[k]]` -- which is the map below.
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> results;
    for (int64_t p : perm)
      results.push_back(getAffineDimExpr(p, ctx));
    AffineMap outMap = AffineMap::get(inTy.getRank(), 0, results, ctx);

    SmallVector<AffineMap> fused(maps.begin(), maps.end() - 1);
    fused.push_back(outMap);

    Location loc = transpose.getLoc();
    auto fusedOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{outTy}, producer.getInputs(),
        ValueRange{transpose.getInit()}, fused,
        producer.getIteratorTypesArray());
    rewriter.cloneRegionBefore(producer.getRegion(), fusedOp.getRegion(),
                               fusedOp.getRegion().end());
    rewriter.replaceOp(transpose, fusedOp.getResults());
    return success();
  }
};

class RequantizeByOneIsACopy : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity())
      return failure();
    Value in = generic.getInputs()[0];
    if (in.getType() != generic->getResult(0).getType() ||
        !getElementTypeOrSelf(in.getType()).isInteger(8))
      return failure();

    Block &body = generic.getRegion().front();
    if (!body.getArgument(1).use_empty())
      return failure();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    auto trunc = yield.getOperand(0).getDefiningOp<arith::TruncIOp>();
    if (!trunc || !trunc.getType().isInteger(8))
      return failure();

    // The clamp has to be inert: anything narrower than i8 is a real operation.
    Value clamped = trunc.getIn();
    int64_t lo = 0, hi = 0;
    bool sawLo = false, sawHi = false;
    while (true) {
      APInt c;
      if (auto min = clamped.getDefiningOp<arith::MinSIOp>()) {
        if (!matchPattern(min.getRhs(), m_ConstantInt(&c)))
          return failure();
        hi = c.getSExtValue();
        sawHi = true;
        clamped = min.getLhs();
        continue;
      }
      if (auto max = clamped.getDefiningOp<arith::MaxSIOp>()) {
        if (!matchPattern(max.getRhs(), m_ConstantInt(&c)))
          return failure();
        lo = c.getSExtValue();
        sawLo = true;
        clamped = max.getLhs();
        continue;
      }
      break;
    }
    if (!sawLo || !sawHi || lo > -128 || hi < 127)
      return failure();

    auto toInt = clamped.getDefiningOp<arith::FPToSIOp>();
    if (!toInt)
      return failure();
    auto round = toInt.getIn().getDefiningOp<math::RoundEvenOp>();
    if (!round)
      return failure();

    auto constant = [](Value c, double *out) {
      llvm::APFloat f(0.0f);
      if (!matchPattern(c, m_ConstantFloat(&f)))
        return false;
      *out = f.convertToDouble();
      return true;
    };
    // The two sides of the ratio are kept apart rather than divided as they are
    // found: the chain is walked from the bottom, so a scale of `c` over `c`
    // would be computed as `(1/c)*c`, which for most `c` is not 1.
    double numerator = 1.0, denominator = 1.0;
    Value v = round.getOperand();
    while (true) {
      double c = 0.0;
      if (auto mul = v.getDefiningOp<arith::MulFOp>()) {
        if (constant(mul.getRhs(), &c))
          v = mul.getLhs();
        else if (constant(mul.getLhs(), &c))
          v = mul.getRhs();
        else
          return failure();
        if (!(c > 0.0))
          return failure();
        numerator *= c;
        continue;
      }
      if (auto div = v.getDefiningOp<arith::DivFOp>()) {
        if (!constant(div.getRhs(), &c) || !(c > 0.0))
          return failure();
        denominator *= c;
        v = div.getLhs();
        continue;
      }
      break;
    }
    if (numerator != denominator)
      return failure();
    auto widen = v.getDefiningOp<arith::SIToFPOp>();
    if (!widen || widen.getIn() != body.getArgument(0))
      return failure();

    rewriter.replaceOp(generic, in);
    return success();
  }
};

/// `elementwise(collapse_shape(x))` is `collapse_shape(elementwise(x))`.
///
/// The mirror of the pattern above, and the one a classifier needs: the flatten
/// in front of it sits between a convolution's tail and the quantization of its
/// result, so the quantization never gets next to the convolution and the layer
/// stays in software. Moving it back across the reshape lets it fuse with the
/// dequantize, which is the form `--convert-linalg-to-gemmlir` folds into the
/// accelerator call.
class MoveElementwiseBeforeCollapse : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    auto collapse = generic.getInputs()[0].getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse || !collapse->hasOneUse())
      return failure();
    if (!generic.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(), linalg::isParallelIterator))
      return failure();
    if (!llvm::all_of(generic.getIndexingMapsArray(),
                      [](AffineMap m) { return m.isIdentity(); }))
      return failure();
    if (!generic.getRegion().front().getArguments().back().use_empty())
      return failure();

    auto srcTy = llvm::dyn_cast<RankedTensorType>(collapse.getSrc().getType());
    auto resTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!srcTy || !resTy || !srcTy.hasStaticShape() || !resTy.hasStaticShape())
      return failure();

    Location loc = generic.getLoc();
    unsigned rank = srcTy.getRank();
    auto wideTy = RankedTensorType::get(srcTy.getShape(), resTy.getElementType());
    SmallVector<AffineMap> maps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    Value init = rewriter.create<tensor::EmptyOp>(loc, wideTy.getShape(),
                                                  resTy.getElementType());
    auto moved = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{wideTy}, ValueRange{collapse.getSrc()}, ValueRange{init},
        maps, iters);
    rewriter.cloneRegionBefore(generic.getRegion(), moved.getRegion(),
                               moved.getRegion().begin());
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        generic, resTy, moved.getResult(0), collapse.getReassociationIndices());
    return success();
  }
};

/// True when the first thing the consumer does with this operand is widen it.
///
/// An i8 activation is where a layer ends, and whatever follows it begins by
/// converting it back to something wider. That conversion belongs on the far
/// side of the accelerator's own output, so the producing call can still fold
/// its requantization into the mvout.
static bool widensTheOperand(linalg::LinalgOp consumer, OpOperand *operand) {
  if (!consumer.getBlock())
    return false;
  BlockArgument arg = consumer.getMatchingBlockArgument(operand);
  if (!arg)
    return false;
  for (Operation *user : arg.getUsers())
    if (llvm::isa<arith::SIToFPOp, arith::UIToFPOp, arith::ExtSIOp,
                  arith::ExtUIOp>(user))
      return true;
  return false;
}

class FuseElementwiseAroundMatmul
    : public impl::FuseElementwiseAroundMatmulBase<FuseElementwiseAroundMatmul> {
public:
  using impl::FuseElementwiseAroundMatmulBase<
      FuseElementwiseAroundMatmul>::FuseElementwiseAroundMatmulBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    // Fusing a quantization into the matmul that consumes it hides the matmul
    // from --convert-linalg-to-gemmlir, and the operation stops being offloaded
    // at all. Measured: the stock --linalg-fuse-elementwise-ops took a CNN from
    // four offloaded matmuls to one and from 35 ms to 232 ms.
    auto worthFusing = [](OpOperand *fusedOperand) {
      Operation *consumer = fusedOperand->getOwner();
      if (llvm::isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::MatvecOp,
                    linalg::VecmatOp>(consumer))
        return false;
      auto consumerOp = llvm::dyn_cast<linalg::LinalgOp>(consumer);
      if (!consumerOp)
        return true;
      if (linalg::isaContractionOpInterface(consumerOp))
        return false;

      // Fusion copies the producer into the consumer, so a producer with more
      // than one consumer gets computed more than once. A residual block is
      // exactly that shape -- the block's input feeds both the first
      // convolution and the shortcut -- and fusing across it recomputed a whole
      // layer's tail *and* left two convolutions in a four-operand operation
      // that the accelerator matcher could not read. One use only.
      if (!fusedOperand->get().hasOneUse())
        return false;

      // An i8 activation is where a layer ends: it is what the accelerator
      // writes and what the next layer reads. Fusing it into the dequantization
      // that follows puts the widening *inside* the producer, so the producing
      // convolution's tail no longer ends in a requantization and
      // --convert-linalg-to-gemmlir cannot fold it -- the convolution stays a
      // scalar loop to save one pass over the activation. Only this direction
      // is refused; fusing f32 work *into* a quantization is the whole point of
      // this pass.
      //
      // Asking whether the *consumer's result* is f32 is the same question for
      // a convolution network, where the dequantization is a tail of its own.
      // It is not the same question for a **transformer**: there the whole of a
      // GELU and the next layer's requantization are one generic that reads an
      // i8 and yields an i8, so the result type says f32 nowhere and the
      // widening went in anyway. Twelve of a ViT's matmuls kept their i32
      // accumulator and a 17x768 f32 pass over it for that reason. What
      // decides is what the body does to the operand, not the type it ends at.
      auto narrow = llvm::dyn_cast<IntegerType>(
          getElementTypeOrSelf(fusedOperand->get().getType()));
      if (narrow && narrow.getWidth() <= 8 &&
          (llvm::isa<FloatType>(getElementTypeOrSelf(
               consumerOp->getResult(0).getType())) ||
           widensTheOperand(consumerOp, fusedOperand)))
        return false;

      // Nor into a consumer that reads it permuted. The accelerator writes its
      // output in order, so a requantization that also transposes is not one
      // `matchRequantize` can fold -- the matmul would keep its whole tail as
      // an f32 pass over the accumulator. `SplitTransposeOutOfElementwise`
      // takes such a pair apart on purpose; this keeps the next fusion round
      // from putting it back together. A transformer's `K.T` is exactly this
      // shape.
      if (narrow && narrow.getWidth() <= 8) {
        AffineMap map = consumerOp.getMatchingIndexingMap(fusedOperand);
        if (map.isPermutation() && !map.isIdentity())
          return false;
      }

      // Fusion runs the producer's body once per *consumer* iteration. When the
      // consumer iterates more than the producer has elements it reads each of
      // them several times -- an im2col gather reads a 3x3 neighbourhood, so
      // nine -- and fusing recomputes the producer that many times over. That
      // is how a relu on 2048 elements became a relu on 3528 here, and it also
      // pins the conversion on the wrong side of the gather, where
      // --hoist-elementwise-before-gather can no longer move it.
      SmallVector<int64_t> loops = consumerOp.getStaticLoopRanges();
      int64_t iterations = 1;
      for (int64_t r : loops) {
        if (ShapedType::isDynamic(r))
          return true;   // cannot compare; leave the decision alone
        iterations *= r;
      }
      auto produced = llvm::dyn_cast<RankedTensorType>(fusedOperand->get().getType());
      if (!produced || !produced.hasStaticShape())
        return true;
      return iterations <= produced.getNumElements();
    };

    RewritePatternSet patterns(&getContext());
    linalg::populateElementwiseOpsFusionPatterns(patterns, worthFusing);
    patterns.add<SinkCollapseIntoBroadcast, CollapseElementwiseOverExpand,
                 MoveElementwiseBeforeCollapse, FoldConstantOperand,
                 RequantizeByOneIsACopy, DequantizeThenQuantizeIsACopy,
                 SinkElementwiseIntoTranspose>(&getContext());
    // The input's quantization only exists by this point, and it sits on the
    // one transpose the layout rewrite had nowhere to cancel against.
    populateAbsorbTransposePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
