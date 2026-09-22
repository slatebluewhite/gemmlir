//===- OrderLoopsForLocalityPass.cpp ----------------------------*- C++ -*-===//
//
// Puts the cheapest axis innermost in an elementwise loop nest.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_ORDERLOOPSFORLOCALITY
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// How much more a scattered **read** costs than a scattered write.
///
/// Measured, not assumed, and it is the direction that surprises: a store on
/// this core is buffered and does not stall, a load does. Weighting them
/// equally cost `atr` and `atrn` 1.3 ms each, and weighting the write more cost
/// the same. Weighting the read is what keeps a relayout reading in order.
constexpr int64_t kReadWeight = 4;

/// How many bytes an operand's address moves when iteration dimension `d`
/// advances by one, or nothing when the map is not a plain projection.
static std::optional<int64_t> byteStride(AffineMap map, MemRefType type,
                                         unsigned d) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      llvm::any_of(strides, ShapedType::isDynamic))
    return std::nullopt;
  if (!type.getElementType().isIntOrFloat())
    return std::nullopt;
  int64_t elemBytes = type.getElementType().getIntOrFloatBitWidth() / 8;
  if (elemBytes == 0)
    return std::nullopt;

  int64_t total = 0;
  for (auto [r, expr] : llvm::enumerate(map.getResults())) {
    if (auto dim = llvm::dyn_cast<AffineDimExpr>(expr)) {
      if (dim.getPosition() == d)
        total += strides[r] * elemBytes;
      continue;
    }
    if (llvm::isa<AffineConstantExpr>(expr))
      continue;
    return std::nullopt;
  }
  return total;
}

/// An elementwise loop nest is free to run its axes in any order, and the order
/// decides how many cache lines it touches.
///
/// A quantization that also relayouts reads NCHW and writes NHWC, and
/// bufferization leaves the loops in the *destination's* order -- so the
/// innermost axis is the channel, which on the source is 1024 bytes apart. Every
/// load is then its own cache line, used for four of its sixty-four bytes. With
/// the image's width innermost instead the source is contiguous and the
/// destination moves eight bytes a step: both sides use whole lines.
///
/// Putting the axis with the smallest total byte stride innermost is the
/// ordinary rule and it is what this does. Permuting parallel iterators is
/// always legal, so nothing here has to be checked beyond reading the strides.
class OrderForLocality : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.hasPureBufferSemantics())
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    unsigned loops = generic.getNumLoops();
    if (loops < 2)
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != generic->getNumOperands())
      return failure();

    // What one step of each axis costs, summed over the operands -- with the
    // *sources* counted several times over, because a scattered load stalls
    // and a scattered store does not.
    unsigned firstInit = generic.getNumDpsInputs();
    SmallVector<int64_t> cost(loops, 0);
    for (auto [k, operand] : llvm::enumerate(generic->getOperands())) {
      auto type = llvm::dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.hasStaticShape())
        return failure();
      int64_t weight = k >= firstInit ? 1 : kReadWeight;
      for (unsigned d = 0; d < loops; d++) {
        std::optional<int64_t> s = byteStride(maps[k], type, d);
        if (!s)
          return failure();
        cost[d] += weight * (*s < 0 ? -*s : *s);
      }
    }

    // An axis of extent one costs nothing and moves nothing; leave it where it
    // is rather than shuffling it inwards.
    SmallVector<int64_t> extent = generic.getStaticLoopRanges();
    if (extent.size() != loops ||
        llvm::any_of(extent, ShapedType::isDynamic))
      return failure();

    SmallVector<unsigned> order(loops);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
      if ((extent[a] == 1) != (extent[b] == 1))
        return extent[a] == 1;   // unit axes outermost, they cost nothing
      return cost[a] > cost[b];  // biggest stride outermost
    });
    if (llvm::all_of(llvm::enumerate(order),
                     [](auto p) { return p.index() == p.value(); }))
      return failure();

    // `order[i]` is the old axis that becomes loop `i`, so the map from the new
    // iteration space to the old one sends new `i` to old `order[i]`.
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> toOld(loops);
    for (auto [newPos, oldDim] : llvm::enumerate(order))
      toOld[oldDim] = getAffineDimExpr(newPos, ctx);
    AffineMap reorder = AffineMap::get(loops, 0, toOld, ctx);

    SmallVector<AffineMap> newMaps;
    for (AffineMap m : maps)
      newMaps.push_back(m.compose(reorder));

    rewriter.modifyOpInPlace(generic, [&] {
      generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newMaps));
    });
    return success();
  }
};

/// A relayout written as `linalg.transpose` gets no say in its loop order.
///
/// The named op carries only a permutation, and `--convert-linalg-to-loops`
/// walks its iteration space in the **destination's** order -- so the write is
/// contiguous and the read jumps a channel plane every element. GoogLeNet's
/// 1x480x12x12 relayout reads 576 bytes apart, one cache line per four bytes
/// used, and it is the hottest basic block in the model.
///
/// Which side to favour is settled and it is the read (`kReadWeight`), but
/// `OrderForLocality` only looks at `linalg.generic`. Writing the transpose out
/// as one -- the permutation on the read map, identity on the write, a body
/// that yields its argument -- is the same operation and puts it in front of
/// that judgement. It is done here rather than earlier because everything that
/// matches a relayout by name has already run by this point in the pipeline.
///
/// **Tiling it further buys nothing -- measured and reverted.** The cache-line
/// count says it should: holding a 16-channel tile takes GoogLeNet's
/// `1x480x12x12` relayout from 13 lines per 12 elements to 28 per 192. A
/// `--tile-relayout-loops` pass that strip-mined the two axes with
/// `tilePerfectlyNested` read **3265.59 -> 3264.59 ms across the model set,
/// -0.03%**, with six of thirteen slightly *worse*; `vit_tiny` -0.9% and
/// `googlenet` -0.2% were the only movement, and all thirteen were
/// byte-identical against the CPU reference and the previous build.
///
/// The model was wrong about which misses are cold. With the channel axis
/// second-innermost the twelve scattered output lines are **768 bytes and are
/// revisited on each of the sixteen channels**, so they are already resident --
/// the cache was tiling it. What is left is the compulsory traffic of moving
/// 276 KB, which no loop structure removes.
class TransposeToGeneric : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    if (!transpose.hasPureBufferSemantics())
      return failure();
    ArrayRef<int64_t> perm = transpose.getPermutation();
    unsigned rank = perm.size();
    if (rank < 2)
      return failure();

    // The named op says `out[i] = in[j]` where in-dimension `perm[i]` supplies
    // out-dimension `i`, so reading `in` needs, for its dimension k, the out
    // dimension whose permutation entry is k.
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> read(rank);
    for (auto [i, k] : llvm::enumerate(perm)) {
      if (k < 0 || (unsigned)k >= rank || read[k])
        return failure();
      read[k] = getAffineDimExpr(i, ctx);
    }
    SmallVector<AffineMap> maps{AffineMap::get(rank, 0, read, ctx),
                                AffineMap::getMultiDimIdentityMap(rank, ctx)};
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        transpose, TypeRange{}, transpose.getInput(), transpose.getInit(), maps,
        iters, [](OpBuilder &b, Location l, ValueRange args) {
          b.create<linalg::YieldOp>(l, args[0]);
        });
    return success();
  }
};

class OrderLoopsForLocality
    : public impl::OrderLoopsForLocalityBase<OrderLoopsForLocality> {
public:
  using impl::OrderLoopsForLocalityBase<
      OrderLoopsForLocality>::OrderLoopsForLocalityBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TransposeToGeneric, OrderForLocality>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
