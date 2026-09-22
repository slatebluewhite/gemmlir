//===- HoldPerChannelInRegistersPass.cpp -------------------------*- C++ -*-===//
//
// The channel is the innermost loop, so a per-channel number is reloaded every
// element. Move the loop.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_HOLDPERCHANNELINREGISTERS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// Where an operand's map sends each of its dimensions: either an iteration
/// dimension, or the constant zero a unit axis becomes
/// ([[mlir-unit-axis-is-a-constant-zero]]).
///
/// `-1` means "the constant zero". Anything else -- a sum, a stride, a
/// modulus -- and this pass does not touch the operation.
static bool axesOf(AffineMap map, SmallVectorImpl<int64_t> &axes) {
  for (AffineExpr e : map.getResults()) {
    if (auto d = llvm::dyn_cast<AffineDimExpr>(e)) {
      axes.push_back((int64_t)d.getPosition());
      continue;
    }
    auto c = llvm::dyn_cast<AffineConstantExpr>(e);
    if (!c || c.getValue() != 0)
      return false;
    axes.push_back(-1);
  }
  return true;
}

static int64_t bytesOf(MemRefType ty) {
  int64_t n = ty.getElementType().getIntOrFloatBitWidth() / 8;
  for (int64_t d : ty.getShape())
    n *= d;
  return n;
}

/// True when the innermost memory dimension steps by one element, so eight
/// consecutive channels are eight consecutive bytes.
static bool innermostIsPacked(MemRefType ty) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(ty.getStridesAndOffset(strides, offset)) || strides.empty())
    return false;
  return strides.back() == 1;
}

/// DenseNet's integer batch norm reloads `M[c]` and `N[c]` on every element,
/// because NHWC puts the channel innermost and nothing in this pipeline hoists
/// a loop-invariant load. Eight channels at a time become the outermost loop
/// and their coefficients are read once.
///
/// The guards are the board's:
///   * the picture is re-read once per channel group, so it has to fit the L1;
///   * sixteen loads need enough pixels to amortize them;
///   * and one runtime `f32` a channel is not worth it at all -- EfficientNet's
///     gate waits on its convert-multiply-convert chain, not on the load.
class HoldPerChannel : public OpRewritePattern<linalg::GenericOp> {
public:
  HoldPerChannel(MLIRContext *ctx, unsigned lanes, unsigned mostBytes,
                 unsigned leastPixels)
      : OpRewritePattern(ctx), lanes(lanes), mostBytes(mostBytes),
        leastPixels(leastPixels) {}

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.hasPureBufferSemantics() || generic.getOutputs().size() != 1 ||
        generic.getInputs().empty())
      return failure();
    for (utils::IteratorType it : generic.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();

    unsigned loops = generic.getNumLoops();
    if (loops < 2)
      return failure();
    unsigned ch = loops - 1;   // the innermost loop is the one to move out

    SmallVector<int64_t> extent = generic.getStaticLoopRanges();
    if (extent.size() != loops || llvm::any_of(extent, ShapedType::isDynamic))
      return failure();
    int64_t channels = extent[ch];
    if (channels % (int64_t)lanes != 0 || channels < 2 * (int64_t)lanes)
      return failure();
    int64_t pixels = 1;
    for (unsigned d = 0; d < ch; d++)
      pixels *= extent[d];
    if (pixels < (int64_t)leastPixels)
      return failure();

    // The body must not read the output: an accumulator is a different
    // operation and a different question.
    Block &body = generic.getRegion().front();
    if (!body.getArguments().back().use_empty())
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    SmallVector<SmallVector<int64_t>> axes(maps.size());
    for (auto [i, m] : llvm::enumerate(maps))
      if (!axesOf(m, axes[i]))
        return failure();

    // Each operand is one of two shapes: a picture, whose innermost dimension
    // is the channel and which the loop walks; or a per-channel number, whose
    // map names the channel and nothing else.
    SmallVector<unsigned> perChannel, picture;
    unsigned n = generic->getNumOperands();
    int64_t widest = 0;
    bool anyIntegerCoefficient = false;
    for (unsigned i = 0; i < n; i++) {
      auto ty = llvm::dyn_cast<MemRefType>(generic->getOperand(i).getType());
      if (!ty || !ty.hasStaticShape() || !innermostIsPacked(ty))
        return failure();
      bool namesOther = false, namesChannel = false;
      for (int64_t a : axes[i]) {
        if (a == (int64_t)ch) namesChannel = true;
        else if (a >= 0) namesOther = true;
      }
      if (!namesChannel)
        return failure();          // invariant in the channel: not this shape
      if (namesOther || i == n - 1) {
        if (axes[i].back() != (int64_t)ch)
          return failure();        // the channel has to be the fastest axis
        picture.push_back(i);
        widest = std::max(widest, bytesOf(ty));
      } else {
        perChannel.push_back(i);
        if (ty.getElementType().isIntOrIndex())
          anyIntegerCoefficient = true;
      }
    }
    if (perChannel.empty() || picture.size() < 2)
      return failure();
    // One runtime f32 a channel is not worth the loop: measured -1.5% to +4.4%.
    if (perChannel.size() < 2 && !anyIntegerCoefficient)
      return failure();
    if (widest > (int64_t)mostBytes)
      return failure();

    // Nothing is written until every refusal is behind us.
    Location loc = generic.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, lanes);
    Value bound = rewriter.create<arith::ConstantIndexOp>(loc, channels);

    auto group = rewriter.create<scf::ForOp>(loc, zero, bound, step);
    rewriter.setInsertionPointToStart(group.getBody());
    Value g = group.getInductionVar();

    // The lane offsets, and the channel index each lane reads.
    SmallVector<Value> lane(lanes), chIdx(lanes);
    for (unsigned k = 0; k < lanes; k++) {
      lane[k] = rewriter.create<arith::ConstantIndexOp>(loc, k);
      chIdx[k] = k ? rewriter.create<arith::AddIOp>(loc, g, lane[k]).getResult()
                   : g;
    }

    // The coefficients, read once for the whole picture.
    DenseMap<unsigned, SmallVector<Value>> held;
    for (unsigned i : perChannel) {
      SmallVector<Value> vals;
      for (unsigned k = 0; k < lanes; k++) {
        SmallVector<Value> idx;
        for (int64_t a : axes[i])
          idx.push_back(a == (int64_t)ch ? chIdx[k] : zero);
        vals.push_back(rewriter.create<memref::LoadOp>(
            loc, generic->getOperand(i), idx));
      }
      held[i] = vals;
    }

    // The picture, in the order the generic asked for.
    SmallVector<Value> ivs(loops);
    ivs[ch] = Value();
    for (unsigned d = 0; d < ch; d++) {
      // A batch of one is an index, not a loop of one.
      if (extent[d] == 1) {
        ivs[d] = zero;
        continue;
      }
      Value ub = rewriter.create<arith::ConstantIndexOp>(loc, extent[d]);
      auto loop = rewriter.create<scf::ForOp>(loc, zero, ub, one);
      ivs[d] = loop.getInductionVar();
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    for (unsigned k = 0; k < lanes; k++) {
      ivs[ch] = chIdx[k];
      IRMapping map;
      for (unsigned i : picture) {
        if (i == n - 1)
          continue;              // the output is written, not read
        SmallVector<Value> idx;
        for (int64_t a : axes[i])
          idx.push_back(a < 0 ? zero : ivs[a]);
        map.map(body.getArgument(i),
                rewriter.create<memref::LoadOp>(loc, generic->getOperand(i), idx)
                    .getResult());
      }
      for (unsigned i : perChannel)
        map.map(body.getArgument(i), held[i][k]);
      Value yielded;
      for (Operation &op : body.without_terminator())
        rewriter.clone(op, map);
      auto yield = llvm::cast<linalg::YieldOp>(body.getTerminator());
      yielded = map.lookupOrDefault(yield.getOperand(0));
      SmallVector<Value> outIdx;
      for (int64_t a : axes[n - 1])
        outIdx.push_back(a < 0 ? zero : ivs[a]);
      rewriter.create<memref::StoreOp>(loc, yielded,
                                       generic->getOperand(n - 1), outIdx);
    }

    rewriter.setInsertionPoint(generic);
    rewriter.eraseOp(generic);
    return success();
  }

private:
  unsigned lanes, mostBytes, leastPixels;
};

class HoldPerChannelInRegistersPass
    : public impl::HoldPerChannelInRegistersBase<HoldPerChannelInRegistersPass> {
public:
  using impl::HoldPerChannelInRegistersBase<
      HoldPerChannelInRegistersPass>::HoldPerChannelInRegistersBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    if (lanes < 2)
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<HoldPerChannel>(&getContext(), lanes, mostBytes, leastPixels);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
