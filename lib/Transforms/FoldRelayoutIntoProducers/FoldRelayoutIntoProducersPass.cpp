//===- FoldRelayoutIntoProducersPass.cpp -------------------------*- C++ -*-===//
//
// A relayout of a buffer several operations filled in pieces.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FOLDRELAYOUTINTOPRODUCERS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// A `linalg.transpose` of a temporary that nothing else reads, where every
/// operation that filled it wrote one slice, becomes those operations writing
/// the transposed slices directly.
///
/// A grouped convolution's four tails each dequantize their own eight channels
/// into a thirty-two channel NHWC buffer, which is then relaid out to the NCHW
/// the model returns. Each tail already walks its own iteration space; writing
/// the permuted slice instead costs it nothing and the relayout goes away
/// entirely. Measured on `gmin`'s shape, a model's worth of that work:
///
/// | | ms |
/// |---|---|
/// | four tails then one relayout | 6.20 |
/// | four permuting tails | **2.63** |
///
/// That is the whole of why `gmin` costs 25% more than `gmid`, which is the
/// same convolution with the same element count and no relayout.
///
/// **This is not the rewrite that was reverted.** Distributing the transpose
/// over the *join* on tensors left four permuting writes into eight channels of
/// a thirty-two channel buffer -- still interleaved, still strided -- and lost.
/// Done here, after bufferization, each producer writes a contiguous NCHW slab,
/// which is exactly the shape `gmid` already has.
///
/// The slices have to tile the buffer along one axis and cover it: anything
/// less and part of the relayout's result would be left unwritten.
///
/// **One producer covering the whole buffer is a slice too.** DenseNet's stem
/// dequantizes the accelerator's i32 output into an NCHW buffer and the very
/// next operation transposes it straight back to NHWC -- two permutations that
/// cancel, a 256 KB buffer written and read for nothing, and both loops walking
/// a 256-byte stride. PC sampling put the pair at **16% of DenseNet**, the
/// largest block in `forward`. Requiring a `memref.subview` refused it: the
/// producer writes the buffer directly, so its user is not a subview at all.
///
/// | | ms | |
/// |---|---|---|
/// | `densenet121` | 785.55 -> **744.07** | -5.3% |
/// | `vit_tiny` | 361.66 -> **358.97** | -0.7% |
/// | the set | 2672.95 -> **2628.36** | -1.7% |
///
/// Byte for byte against both the CPU reference and the previous build. The
/// object is **45 instructions smaller** on DenseNet and 41.5 ms faster, which
/// is the usual reminder that a count measures nothing here.
class FoldRelayout : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    auto srcTy = llvm::dyn_cast<MemRefType>(transpose.getInput().getType());
    auto dstTy = llvm::dyn_cast<MemRefType>(transpose.getInit().getType());
    if (!srcTy || !dstTy || !srcTy.hasStaticShape() || !dstTy.hasStaticShape() ||
        !srcTy.getLayout().isIdentity() || !dstTy.getLayout().isIdentity())
      return failure();
    auto srcAlloc = transpose.getInput().getDefiningOp<memref::AllocOp>();
    auto dstAlloc = transpose.getInit().getDefiningOp<memref::AllocOp>();
    // Both have to be temporaries this pass can account for completely: the
    // source because every write to it is about to move, the destination
    // because those writes are about to happen earlier than they did.
    if (!srcAlloc || !dstAlloc)
      return failure();

    ArrayRef<int64_t> perm = transpose.getPermutation();
    int64_t rank = srcTy.getRank();
    if ((int64_t)perm.size() != rank)
      return failure();

    // Every use of the source: the transpose, its deallocation, and the
    // operations that filled it -- each writing a slice, or **one** writing the
    // whole thing.
    SmallVector<memref::SubViewOp> slices;
    SmallVector<linalg::GenericOp> producers;
    memref::DeallocOp dealloc;
    for (Operation *user : srcAlloc->getUsers()) {
      if (user == transpose.getOperation())
        continue;
      if (auto d = llvm::dyn_cast<memref::DeallocOp>(user)) {
        dealloc = d;
        continue;
      }
      // One producer writing the buffer entire. DenseNet's stem is this: the
      // accelerator's i32 output is dequantized into an NCHW buffer and the
      // very next operation transposes it back to NHWC, so the two
      // permutations cancel and a 256 KB buffer stops existing. Requiring a
      // `memref.subview` refused it -- the degenerate case of "the slices tile
      // one axis and cover it" is one slice that is the whole axis.
      if (auto producer = llvm::dyn_cast<linalg::GenericOp>(user)) {
        if (producer.getOutputs().size() != 1 ||
            producer.getOutputs()[0] != srcAlloc.getResult() ||
            !producer.getIndexingMapsArray().back().isProjectedPermutation())
          return failure();
        if (llvm::is_contained(producer.getInputs(), srcAlloc.getResult()))
          return failure();
        slices.push_back(nullptr);
        producers.push_back(producer);
        continue;
      }
      auto slice = llvm::dyn_cast<memref::SubViewOp>(user);
      if (!slice || !slice->hasOneUse())
        return failure();
      auto producer =
          llvm::dyn_cast<linalg::GenericOp>(*slice->getUsers().begin());
      if (!producer || producer.getOutputs().size() != 1 ||
          producer.getOutputs()[0] != slice.getResult() ||
          !producer.getIndexingMapsArray().back().isProjectedPermutation())
        return failure();
      slices.push_back(slice);
      producers.push_back(producer);
    }
    if (slices.empty())
      return failure();
    // A whole-buffer producer is the only one there can be.
    if (llvm::is_contained(slices, memref::SubViewOp()) && slices.size() != 1)
      return failure();

    // The slices have to tile one axis and cover the buffer. They do **not**
    // have to be the same width: a grouped convolution's four tails are, but an
    // Inception block's four branches are 128, 192, 96 and 64 channels, and
    // requiring equal slices refused both of GoogLeNet's relayouts -- 17.7% of
    // the model in a pure copy.
    int64_t axis = -1;
    SmallVector<SmallVector<int64_t>> allSizes, allOffs;
    SmallVector<int64_t> offsets;
    for (memref::SubViewOp slice : slices) {
      SmallVector<int64_t> off(rank, 0), size(srcTy.getShape());
      if (slice) {
        if (!slice.getOffsets().empty() || !slice.getSizes().empty() ||
            !slice.getStrides().empty())
          return failure();
        off.assign(slice.getStaticOffsets().begin(),
                   slice.getStaticOffsets().end());
        size.assign(slice.getStaticSizes().begin(),
                    slice.getStaticSizes().end());
        if (llvm::any_of(off, ShapedType::isDynamic) ||
            llvm::any_of(size, ShapedType::isDynamic))
          return failure();
        for (int64_t s : slice.getStaticStrides())
          if (s != 1)
            return failure();
      }
      // The axis is the one a slice is narrower on, or the one it is moved
      // along; with unequal widths the first slice may be at offset zero and
      // still be the one that names it.
      for (int64_t d = 0; d < rank; d++) {
        if (off[d] == 0 && size[d] == srcTy.getShape()[d])
          continue;
        if (axis >= 0 && axis != d)
          return failure();
        axis = d;
      }
      allOffs.push_back(off);
      allSizes.push_back(size);
      offsets.push_back(0);
    }
    if (axis < 0)
      axis = rank - 1;
    for (auto [k, off] : llvm::enumerate(allOffs)) {
      offsets[k] = off[axis];
      for (int64_t d = 0; d < rank; d++)
        if (d != axis &&
            (off[d] != 0 || allSizes[k][d] != srcTy.getShape()[d]))
          return failure();
    }
    // Sorted by offset they have to start at zero, meet exactly, and finish at
    // the end -- anything less and part of the relayout's result would be left
    // unwritten.
    SmallVector<unsigned> order(slices.size());
    std::iota(order.begin(), order.end(), 0u);
    llvm::sort(order, [&](unsigned a, unsigned b) {
      return offsets[a] < offsets[b];
    });
    int64_t reach = 0;
    for (unsigned k : order) {
      if (offsets[k] != reach || allSizes[k][axis] <= 0)
        return failure();
      reach += allSizes[k][axis];
    }
    if (reach != srcTy.getShape()[axis])
      return failure();

    // The destination is written where the producers are now, so it has to
    // exist by then.
    Operation *earliest = producers.front();
    for (linalg::GenericOp producer : producers)
      if (producer->isBeforeInBlock(earliest))
        earliest = producer;
    if (dstAlloc->isBeforeInBlock(earliest) == false)
      rewriter.moveOpBefore(dstAlloc, earliest);

    MLIRContext *ctx = transpose.getContext();
    for (auto [k, pair] : llvm::enumerate(llvm::zip(slices, producers))) {
      auto [slice, producer] = pair;
      rewriter.setInsertionPoint(producer);
      Value into = dstAlloc.getResult();
      if (slice) {
        ArrayRef<int64_t> off = slice.getStaticOffsets();
        SmallVector<OpFoldResult> newOff, newSize, newStride;
        for (int64_t d = 0; d < rank; d++) {
          newOff.push_back(rewriter.getIndexAttr(off[perm[d]]));
          newSize.push_back(rewriter.getIndexAttr(allSizes[k][perm[d]]));
          newStride.push_back(rewriter.getIndexAttr(1));
        }
        into = rewriter.create<memref::SubViewOp>(
            slice.getLoc(), dstAlloc.getResult(), newOff, newSize, newStride);
      }

      SmallVector<AffineMap> maps = producer.getIndexingMapsArray();
      AffineMap out = maps.back();
      SmallVector<AffineExpr> results;
      for (int64_t d = 0; d < rank; d++)
        results.push_back(out.getResult(perm[d]));
      maps.back() =
          AffineMap::get(out.getNumDims(), out.getNumSymbols(), results, ctx);

      rewriter.modifyOpInPlace(producer, [&] {
        producer.getOutputsMutable().assign(into);
        producer.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
      });
      if (slice)
        rewriter.eraseOp(slice);
    }

    rewriter.eraseOp(transpose);
    if (dealloc)
      rewriter.eraseOp(dealloc);
    rewriter.eraseOp(srcAlloc);
    return success();
  }
};

class FoldRelayoutIntoProducers
    : public impl::FoldRelayoutIntoProducersBase<FoldRelayoutIntoProducers> {
public:
  using impl::FoldRelayoutIntoProducersBase<
      FoldRelayoutIntoProducers>::FoldRelayoutIntoProducersBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldRelayout>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
