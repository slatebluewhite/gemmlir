//===- PoolWithoutPaddingPass.cpp --------------------------------*- C++ -*-===//
//
// A padded pool is a few unpadded pools on bands of its output.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_POOLWITHOUTPADDING
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// A run of output indices whose window is clipped by the padding in the same
/// way, described by the range of kernel taps that land on the real image.
struct Band {
  int64_t oLo, oHi;  // output range, half open
  int64_t tLo, tHi;  // kernel taps that are not padding, inclusive
};

/// The bands of one spatial axis.
///
/// Output index `o` reads the padded image at `o*s + t*d` for tap `t`, which is
/// the real image at `o*s - lo + t*d`. A tap is padding when that falls outside
/// `[0, extent)`. Since the window slides by a constant stride, the taps that
/// are clipped change only near the two edges, and each distinct clipping holds
/// over a contiguous run of `o` -- which is the band.
///
/// Empty when some output reads nothing but padding: that answer is the pad
/// value alone and this pass does not build it.
static SmallVector<Band> bandsFor(int64_t out, int64_t k, int64_t s, int64_t d,
                                  int64_t lo, int64_t extent) {
  SmallVector<Band> bands;
  for (int64_t o = 0; o < out; o++) {
    int64_t tLo = 0, tHi = k - 1;
    while (tLo <= tHi && o * s - lo + tLo * d < 0)
      tLo++;
    while (tHi >= tLo && o * s - lo + tHi * d > extent - 1)
      tHi--;
    if (tLo > tHi)
      return {};
    if (!bands.empty() && bands.back().tLo == tLo && bands.back().tHi == tHi &&
        bands.back().oHi == o)
      bands.back().oHi = o + 1;
    else
      bands.push_back({o, o + 1, tLo, tHi});
  }
  return bands;
}

/// The constant a fill writes, whichever of the two forms it takes.
static Value filledWith(Operation *op, Value buffer) {
  if (auto fill = llvm::dyn_cast<linalg::FillOp>(op)) {
    if (fill.hasPureBufferSemantics() && fill.getInputs().size() == 1 &&
        fill.getOutputs().size() == 1 && fill.getOutputs()[0] == buffer)
      return fill.getInputs()[0];
    return nullptr;
  }
  auto map = llvm::dyn_cast<linalg::MapOp>(op);
  if (!map || !map.hasPureBufferSemantics() || map.getInputs().size() != 0 ||
      map.getInit() != buffer)
    return nullptr;
  auto yield =
      llvm::dyn_cast<linalg::YieldOp>(map.getMapper().front().getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return nullptr;
  Value v = yield.getOperand(0);
  return v.getParentBlock() == &map.getMapper().front() ? nullptr : v;
}

/// A pool over a padded copy of a buffer reads a buffer that was filled with a
/// constant and then had the real image copied into the middle of it. GoogLeNet
/// pays **1.3 million element moves an inference** for that -- 773,000 copied
/// and 528,000 filled -- and thirteen of its eighteen copies come straight out
/// of the accelerator, which cannot be asked to write into the middle of a
/// padded buffer because Gemmini's output has one stride, not a row stride and
/// a pixel stride.
///
/// So take the padding away instead. Cut the output into bands on which the
/// clipping is constant and give each band its own pool, reading the real
/// image directly through a subview with a smaller window. For the first of
/// GoogLeNet's pools -- 48x48 into 24x24, 3x3 at stride 2, two rows and columns
/// of `ceil_mode` padding -- that is four regions: the 23x23 interior with the
/// full window, and three edge strips whose window is 3x2, 2x3 and 2x2.
///
/// A band that touches padding has its output filled with the **pad value**
/// first, so its pool computes `max(p, the real taps)`, which is what the
/// padded pool computed. A band that does not keeps whatever identity the
/// output was already initialised with.
///
/// | | ms | |
/// |---|---|---|
/// | `densenet121` | 712.67 -> **672.11** | -5.7% |
/// | `googlenet`   | 341.80 -> **331.00** | -3.2% |
/// | the set       | 2512.43 -> **2461.93** | -2.0% |
///
/// Byte for byte against both references, which is the check that matters here:
/// a band's offsets and window size are arithmetic that has to be exactly
/// right, and getting it wrong gives a plausible wrong answer rather than a
/// crash.
class PoolWithoutPadding : public OpRewritePattern<linalg::PoolingNhwcMaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNhwcMaxOp pool,
                                PatternRewriter &rewriter) const final {
    if (pool.getInputs().size() != 2 || pool.getOutputs().size() != 1 ||
        pool->getNumResults() != 0)
      return failure();
    Value pad = pool.getInputs()[0], window = pool.getInputs()[1];
    Value out = pool.getOutputs()[0];
    auto padTy = llvm::dyn_cast<MemRefType>(pad.getType());
    auto outTy = llvm::dyn_cast<MemRefType>(out.getType());
    auto winTy = llvm::dyn_cast<MemRefType>(window.getType());
    if (!padTy || !outTy || !winTy || padTy.getRank() != 4 ||
        outTy.getRank() != 4 || winTy.getRank() != 2 ||
        !padTy.hasStaticShape() || !outTy.hasStaticShape() ||
        !winTy.hasStaticShape() || !padTy.getLayout().isIdentity())
      return failure();

    // The padded buffer: one constant fill, one copy into a box of it, and this
    // pool. Anything else and the buffer is not simply a padding.
    Operation *fillOp = nullptr;
    Value padValue;
    memref::CopyOp copy;
    memref::SubViewOp box;
    memref::DeallocOp dealloc;
    for (Operation *user : pad.getUsers()) {
      if (user == pool.getOperation())
        continue;
      if (auto d = llvm::dyn_cast<memref::DeallocOp>(user)) {
        if (dealloc)
          return failure();
        dealloc = d;
        continue;
      }
      if (Value v = filledWith(user, pad)) {
        if (fillOp)
          return failure();
        fillOp = user;
        padValue = v;
        continue;
      }
      auto slice = llvm::dyn_cast<memref::SubViewOp>(user);
      if (!slice || !slice->hasOneUse() || copy)
        return failure();
      auto c = llvm::dyn_cast<memref::CopyOp>(*slice->getUsers().begin());
      if (!c || c.getTarget() != slice.getResult())
        return failure();
      copy = c;
      box = slice;
    }
    if (!fillOp || !copy || !box)
      return failure();
    if (!fillOp->isBeforeInBlock(copy) || !copy->isBeforeInBlock(pool))
      return failure();

    if (!box.getOffsets().empty() || !box.getSizes().empty() ||
        !box.getStrides().empty())
      return failure();
    ArrayRef<int64_t> lo = box.getStaticOffsets();
    ArrayRef<int64_t> size = box.getStaticSizes();
    if (llvm::any_of(lo, ShapedType::isDynamic) ||
        llvm::any_of(size, ShapedType::isDynamic))
      return failure();
    for (int64_t s : box.getStaticStrides())
      if (s != 1)
        return failure();
    Value src = copy.getSource();
    auto srcTy = llvm::dyn_cast<MemRefType>(src.getType());
    if (!srcTy || !srcTy.hasStaticShape() || srcTy.getShape() != size)
      return failure();
    // Batch and channel are never padded, and the pool walks them whole.
    if (lo[0] != 0 || lo[3] != 0 || size[0] != padTy.getShape()[0] ||
        size[3] != padTy.getShape()[3])
      return failure();

    auto pair = [](DenseIntElementsAttr a, unsigned i) -> int64_t {
      return (*(a.value_begin<APInt>() + i)).getSExtValue();
    };
    auto strides = pool.getStrides(), dilations = pool.getDilations();
    if (!strides || !dilations || strides.getNumElements() != 2 ||
        dilations.getNumElements() != 2)
      return failure();
    int64_t sh = pair(strides, 0), sw = pair(strides, 1);
    int64_t dh = pair(dilations, 0), dw = pair(dilations, 1);
    int64_t kh = winTy.getShape()[0], kw = winTy.getShape()[1];

    SmallVector<Band> bh = bandsFor(outTy.getShape()[1], kh, sh, dh, lo[1],
                                    srcTy.getShape()[1]);
    SmallVector<Band> bw = bandsFor(outTy.getShape()[2], kw, sw, dw, lo[2],
                                    srcTy.getShape()[2]);
    if (bh.empty() || bw.empty())
      return failure();
    // More than a handful of regions and the loop nests cost more than the
    // copy they save. One band each way is a pool that never reads the padding
    // -- the case `--drop-unread-padding` catches in the frontend -- and is
    // worth doing here anyway, since it still takes the buffer and the copy.
    if (bh.size() * bw.size() > 9)
      return failure();

    // This used to refuse an i8 pool. `--pack-int8-max-pool` puts eight i8
    // channels in one `i64`, which needs a collapse to a flat buffer and so an
    // identity layout; bands write strided subviews, the packing refused them,
    // and the pool went back to a byte at a time -- which cost far more than
    // the copy this saves (`googlenet` 341.74 -> 444.68 ms, +30.1%, against
    // `densenet121`, whose one pool is f32 and never packed, at 712.34 ->
    // 671.54, -5.7%).
    //
    // The packing reads a band now: a rank-preserving, unit-stride subview that
    // cuts only the spatial dimensions is the same buffer with the same eight
    // bytes to a word, starting further in. So both apply, and the copy goes.

    Location loc = pool.getLoc();
    rewriter.setInsertionPoint(pool);
    auto idx = [&](int64_t v) { return rewriter.getIndexAttr(v); };
    SmallVector<OpFoldResult> ones(4, idx(1));
    SmallVector<OpFoldResult> winOnes(2, idx(1)), winZero(2, idx(0));

    for (const Band &h : bh) {
      for (const Band &w : bw) {
        int64_t kbh = h.tHi - h.tLo + 1, kbw = w.tHi - w.tLo + 1;
        SmallVector<OpFoldResult> inOff{
            idx(0), idx(h.oLo * sh - lo[1] + h.tLo * dh),
            idx(w.oLo * sw - lo[2] + w.tLo * dw), idx(0)};
        SmallVector<OpFoldResult> inSize{
            idx(srcTy.getShape()[0]),
            idx((h.oHi - h.oLo - 1) * sh + (kbh - 1) * dh + 1),
            idx((w.oHi - w.oLo - 1) * sw + (kbw - 1) * dw + 1),
            idx(srcTy.getShape()[3])};
        Value in =
            rewriter.create<memref::SubViewOp>(loc, src, inOff, inSize, ones);
        SmallVector<OpFoldResult> outOff{idx(0), idx(h.oLo), idx(w.oLo),
                                         idx(0)};
        SmallVector<OpFoldResult> outSize{idx(outTy.getShape()[0]),
                                          idx(h.oHi - h.oLo),
                                          idx(w.oHi - w.oLo),
                                          idx(outTy.getShape()[3])};
        Value into =
            rewriter.create<memref::SubViewOp>(loc, out, outOff, outSize, ones);
        Value win = rewriter.create<memref::SubViewOp>(
            loc, window, winZero,
            SmallVector<OpFoldResult>{idx(kbh), idx(kbw)}, winOnes);
        // A band that reaches into the padding answers `max(p, the real taps)`.
        if (kbh != kh || kbw != kw)
          rewriter.create<linalg::FillOp>(loc, ValueRange{padValue},
                                          ValueRange{into});
        SmallVector<Value> ins{in, win};
        rewriter.create<linalg::PoolingNhwcMaxOp>(
            loc, TypeRange{}, ins, ValueRange{into}, strides, dilations);
      }
    }

    rewriter.eraseOp(pool);
    rewriter.eraseOp(copy);
    rewriter.eraseOp(box);
    rewriter.eraseOp(fillOp);
    if (dealloc)
      rewriter.eraseOp(dealloc);
    if (Operation *alloc = pad.getDefiningOp())
      if (llvm::isa<memref::AllocOp>(alloc) && alloc->use_empty())
        rewriter.eraseOp(alloc);
    return success();
  }
};

class PoolWithoutPaddingPass
    : public impl::PoolWithoutPaddingBase<PoolWithoutPaddingPass> {
public:
  using impl::PoolWithoutPaddingBase<PoolWithoutPaddingPass>::PoolWithoutPaddingBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PoolWithoutPadding>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
