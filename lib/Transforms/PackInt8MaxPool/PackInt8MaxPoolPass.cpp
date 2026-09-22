//===- PackInt8MaxPoolPass.cpp -----------------------------------*- C++ -*-===//
//
// Eight channels of a max-pool in one 64-bit word.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_PACKINT8MAXPOOL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

constexpr uint64_t kHigh = 0x8080808080808080ull;
constexpr uint64_t kLow = 0x0101010101010101ull;
constexpr int64_t kLanes = 8;

/// The row and column values a pooling op carries as a two-element attribute.
///
/// They used to have to be equal, which was true of every pool a frontend
/// emits. `--separate-max-pool` splits one into a `[1, s]` pass and an `[s, 1]`
/// one, and the packing below already indexes the rows and the columns apart,
/// so there was never a reason to insist.
static bool pairValues(DenseIntElementsAttr a, int64_t &row, int64_t &col) {
  if (!a || a.getNumElements() != 2)
    return false;
  auto it = a.value_begin<APInt>();
  row = (*it).getSExtValue();
  col = (*(it + 1)).getSExtValue();
  return true;
}

static bool isViewLike(Operation *op) {
  return llvm::isa_and_nonnull<memref::SubViewOp, memref::ExpandShapeOp,
                               memref::CollapseShapeOp, memref::CastOp,
                               memref::ViewOp>(op);
}

static bool nonNegativeI8(Value buffer, int depth);

/// Does `user` write through `use`, and if so does it only ever write a byte in
/// [0,127]?  Anything not recognised counts as a write that might be negative.
static bool writesOnlyNonNegative(Operation *user, OpOperand &use, int depth) {
  Value v = use.get();
  if (auto copy = llvm::dyn_cast<memref::CopyOp>(user)) {
    if (copy.getTarget() != v)
      return true; // a read
    return nonNegativeI8(copy.getSource(), depth + 1);
  }
  // The accelerator applies `relu` in the accumulator's scale pipeline, and an
  // i8 output is scaled and saturated on the way out, so a relu'd result is in
  // [0,127]. (The `full_C` caveat in GemmlirOps.td is about an i32 output,
  // which these do not have.)
  if (auto conv = llvm::dyn_cast<Conv2DInt8Op>(user))
    return conv.getOutput() != v || conv.getAct() == Act::RELU;
  if (auto dw = llvm::dyn_cast<DepthwiseConv2DInt8Op>(user))
    return dw.getOutput() != v || dw.getAct() == Act::RELU;
  if (auto mm = llvm::dyn_cast<MatMulInt8ScaleOp>(user))
    return mm.getOutMat() != v || mm.getAct() == Act::RELU;
  // The rest of the dialect: a read of the buffer is no obstacle, and the only
  // question is whether it is the one being written.
  if (auto mm = llvm::dyn_cast<MatMulInt8Op>(user))
    return mm.getOutMat() != v; // an i32 accumulator, never an i8 pool input
  if (auto ra = llvm::dyn_cast<ResAddInt8Op>(user))
    return ra.getOutMat() != v || ra.getAct() == Act::RELU;
  if (auto nm = llvm::dyn_cast<NormInt8Op>(user))
    return nm.getOutput() != v;
  // `--fill-to-memset` turns the padding fills into these; the byte it writes
  // is the one the pool would read.
  // `getValue()` comes back *unsigned*, so -128 reads as 128 and a plain
  // `>= 0` is always true -- the padding byte has to be read as a signed one.
  if (auto memset = llvm::dyn_cast<MemsetOp>(user))
    return memset.getBuffer() != v ||
           static_cast<int8_t>(memset.getValue()) >= 0;
  // A `linalg.fill` keeps its value in `ins` and yields the block argument, so
  // looking only at what the body yields finds an argument, not a constant.
  if (auto fill = llvm::dyn_cast<linalg::FillOp>(user)) {
    if (fill.getOutputs()[0] != v)
      return true;
    IntegerAttr cst;
    return matchPattern(fill.getInputs()[0], m_Constant(&cst)) &&
           cst.getValue().isNonNegative();
  }

  if (auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(user)) {
    bool writes = false;
    for (OpOperand *out : linalgOp.getDpsInitsMutable().empty()
                              ? SmallVector<OpOperand *>{}
                              : llvm::to_vector(llvm::map_range(
                                    linalgOp.getDpsInitsMutable(),
                                    [](OpOperand &o) { return &o; })))
      if (out->get() == v)
        writes = true;
    if (!writes)
      return true; // a read
    // The padding a pool reads is written by a fill, and it has to be a
    // non-negative one -- Gemmini pads with zero and so does this.
    Block *body = linalgOp.getBlock();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return false;
    IntegerAttr cst;
    if (matchPattern(yield.getOperand(0), m_Constant(&cst)))
      return cst.getValue().isNonNegative();
    // A relayout yields what it read.
    if (auto arg = llvm::dyn_cast<BlockArgument>(yield.getOperand(0)))
      if (arg.getOwner() == body &&
          arg.getArgNumber() < linalgOp.getNumDpsInputs())
        return nonNegativeI8(linalgOp.getDpsInputs()[arg.getArgNumber()],
                             depth + 1);
    return false;
  }
  if (llvm::isa<memref::DeallocOp>(user))
    return true;
  return false;
}

/// The `linalg.fill` that set the buffer up, if that is what it was: the first
/// thing going backwards that writes it has to be a fill of the whole of it,
/// and nothing may write a piece of it in between.
static linalg::FillOp fillOf(Value out, Operation *pool) {
  for (Operation *op = pool->getPrevNode(); op; op = op->getPrevNode()) {
    if (isMemoryEffectFree(op))
      continue;
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      // Anything that might write it stops the search; anything that cannot
      // touch it is skipped by the alias test below.
      bool touches = false;
      for (Value operand : op->getOperands())
        if (operand == out)
          touches = true;
      if (touches)
        return nullptr;
      continue;
    }
    bool writes = false;
    for (OpOperand &init : linalgOp.getDpsInitsMutable())
      if (init.get() == out)
        writes = true;
    if (!writes)
      continue;
    return llvm::dyn_cast<linalg::FillOp>(op);
  }
  return nullptr;
}

/// Every byte reachable through `buffer` is in [0,127].
///
/// Safe to walk the buffer's own value rather than its allocation: this runs
/// after `--plan-static-buffers`, which is a bump allocator handing every
/// buffer one `memref.view` at its own offset, so two views are two buffers
/// ([[gemmlir-one-arena-is-one-buffer]] is about going the *other* way, down to
/// the arena, which would make everything alias).
static bool nonNegativeI8(Value buffer, int depth) {
  if (depth > 3)
    return false;
  auto ty = llvm::dyn_cast<MemRefType>(buffer.getType());
  if (!ty || !ty.getElementType().isInteger(8))
    return false;
  SmallVector<Value> work{buffer};
  SmallPtrSet<Operation *, 16> seen;
  bool written = false;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (isViewLike(user)) {
        if (seen.insert(user).second)
          work.push_back(user->getResult(0));
        continue;
      }
      if (!writesOnlyNonNegative(user, use, depth))
        return false;
      written = true;
    }
  }
  return written;
}

/// A max-pool over i8 is **independent per channel**, and NHWC puts the channels
/// next to each other -- so eight of them are one 64-bit word and the whole
/// window can be walked eight at a time.
///
/// The scalar form is what a program-counter profile finds at the top of
/// GoogLeNet: **39% of the model** in nine byte loads and eight compares per
/// output, each compare a data-dependent branch the predictor cannot help with.
/// Measured on the board on one of its shapes (14x14x256 in, 3x3, 12x12x256
/// out), 200 repetitions: **111.10 ms scalar against 21.65 ms packed, 5.13x**.
///
/// Byte-wise unsigned maximum of two words, with every byte flipped into
/// unsigned order once on the way in and once on the way out:
///
/// ```
///   d   = (a | HI) - (b & ~HI)                 // bit 7: al >= bl
///   ge  = HI & ((a & ~b) | (~(a ^ b) & d))     // bit 7: a_i >= b_i
///   m   = ((ge >> 7) & LOW) * 0xFF             // 0xFF per byte where it is
///   max = b ^ ((a ^ b) & m)
/// ```
///
/// Writing a byte as `0x80*ah + al`, `(a | HI) - (b & ~HI)` is `0x80 + al - bl`
/// per byte, which cannot borrow out of its byte, so its bit 7 is `al >= bl`.
/// Unsigned `a_i >= b_i` is then `ah & ~bh`, or `al >= bl` where the high bits
/// agree. **Checked exhaustively**: all 65,536 byte pairs and 200,000 random
/// word pairs against the definition, in `scripts/poolswar.c`. The obvious
/// shorter formula -- `((a|HI) - (b&~HI)) ^ ((a ^ ~b) & HI)` -- is wrong on
/// 248,893 of them, which is why that check is there.
///
/// The accumulator starts from what the output buffer already holds, which is
/// exactly `linalg.pooling_nhwc_max`'s own semantics and means the fill in front
/// of it needs no special case.
///
/// **The index is rebuilt from its induction variables on every step, and that
/// is fine -- measured.** Carrying a partial sum down the nest instead, so the
/// innermost body holds one multiply and one add rather than eight, changed
/// GoogLeNet's `forward` by **six instructions** and the model set by +0.02%.
/// `llc` runs no IR pipeline, but its instruction selection still eliminates
/// redundancy *within a basic block* -- and after `--unroll-reduction-windows`
/// the window is straight-line, so it had already done it. Redundancy across a
/// loop's back edge is the kind worth removing in MLIR
/// ([[gemmlir-the-accumulator-lives-in-memory]]); redundancy inside one body is
/// not.
class PackInt8MaxPool : public OpRewritePattern<linalg::PoolingNhwcMaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

/// A **band** is what `--pool-without-padding` writes: a rank-preserving,
/// unit-stride `memref.subview` that cuts only the spatial dimensions of a
/// contiguous buffer. The bytes are still eight to a word; the band just starts
/// further into the buffer.
///
/// Refusing these is what kept the banding pass away from every pool this one
/// can pack, and with it the padded copy the banding exists to remove --
/// GoogLeNet moves half a million elements an inference into the middle of
/// padded buffers, because the accelerator cannot be asked to write there
/// ([[gemmini-one-stride-per-pixel]]).
struct Band {
  Value buffer;       // the contiguous buffer the band views
  MemRefType type;    // its type, which is where the strides come from
  int64_t baseWord;   // where the band starts, in i64 units
};

static std::optional<Band> bandOf(Value v) {
  auto ty = llvm::dyn_cast<MemRefType>(v.getType());
  if (!ty || !ty.hasStaticShape() || ty.getRank() != 4)
    return std::nullopt;
  if (ty.getLayout().isIdentity())
    return Band{v, ty, 0};
  auto sub = v.getDefiningOp<memref::SubViewOp>();
  if (!sub)
    return std::nullopt;
  auto srcTy = llvm::dyn_cast<MemRefType>(sub.getSource().getType());
  if (!srcTy || !srcTy.hasStaticShape() || !srcTy.getLayout().isIdentity() ||
      srcTy.getRank() != 4)
    return std::nullopt;
  if (!sub.getDroppedDims().none())
    return std::nullopt;
  SmallVector<int64_t> offsets, sizes, steps;
  for (OpFoldResult o : sub.getMixedOffsets()) {
    std::optional<int64_t> c = getConstantIntValue(o);
    if (!c) return std::nullopt;
    offsets.push_back(*c);
  }
  for (OpFoldResult o : sub.getMixedSizes()) {
    std::optional<int64_t> c = getConstantIntValue(o);
    if (!c) return std::nullopt;
    sizes.push_back(*c);
  }
  for (OpFoldResult o : sub.getMixedStrides()) {
    std::optional<int64_t> c = getConstantIntValue(o);
    if (!c || *c != 1) return std::nullopt;
  }
  if (offsets.size() != 4 || sizes.size() != 4)
    return std::nullopt;
  // The channel axis is what the word packing is made of, so a band may not
  // cut it.
  if (offsets[3] != 0 || sizes[3] != srcTy.getDimSize(3))
    return std::nullopt;
  int64_t stride = 1, byteOffset = 0;
  for (int d = 3; d >= 0; d--) {
    byteOffset += offsets[d] * stride;
    stride *= srcTy.getDimSize(d);
  }
  if (byteOffset % kLanes != 0)
    return std::nullopt;
  return Band{sub.getSource(), srcTy, byteOffset / kLanes};
}

  LogicalResult matchAndRewrite(linalg::PoolingNhwcMaxOp pool,
                                PatternRewriter &rewriter) const final {
    if (pool.getInputs().size() != 2 || pool.getOutputs().size() != 1 ||
        pool->getNumResults() != 0)
      return failure();
    Value src = pool.getInputs()[0], out = pool.getOutputs()[0];
    auto srcTy = llvm::dyn_cast<MemRefType>(src.getType());
    auto outTy = llvm::dyn_cast<MemRefType>(out.getType());
    auto winTy = llvm::dyn_cast<MemRefType>(pool.getInputs()[1].getType());
    if (!srcTy || !outTy || !winTy)
      return failure();
    if (srcTy.getRank() != 4 || outTy.getRank() != 4 || winTy.getRank() != 2)
      return failure();
    if (!srcTy.hasStaticShape() || !outTy.hasStaticShape() ||
        !winTy.hasStaticShape())
      return failure();
    if (!srcTy.getElementType().isInteger(8) ||
        !outTy.getElementType().isInteger(8))
      return failure();
    // A view of the bytes as words needs a buffer that starts where it says and
    // is laid out densely: `memref.view` takes an identity `memref<?xi8>`. A
    // band of one is that buffer plus an offset.
    std::optional<Band> srcBand = bandOf(src), outBand = bandOf(out);
    if (!srcBand || !outBand)
      return failure();
    if (srcBand->type.getDimSize(3) != srcTy.getDimSize(3) ||
        outBand->type.getDimSize(3) != outTy.getDimSize(3))
      return failure();

    int64_t channels = srcTy.getDimSize(3);
    if (channels != outTy.getDimSize(3) || channels % kLanes != 0)
      return failure();
    if (srcTy.getDimSize(0) != outTy.getDimSize(0))
      return failure();

    int64_t strideRow = 0, strideCol = 0, dilRow = 0, dilCol = 0;
    if (!pairValues(pool.getStrides(), strideRow, strideCol) ||
        !pairValues(pool.getDilations(), dilRow, dilCol) || strideRow < 1 ||
        strideCol < 1 || dilRow < 1 || dilCol < 1)
      return failure();

    int64_t n = outTy.getDimSize(0), oh = outTy.getDimSize(1),
            ow = outTy.getDimSize(2);
    int64_t ih = srcTy.getDimSize(1), iw = srcTy.getDimSize(2);
    int64_t kh = winTy.getDimSize(0), kw = winTy.getDimSize(1);
    // The op's own shape rule; if it does not hold the window would read out of
    // bounds and this is not the operation it claims to be.
    if ((oh - 1) * strideRow + (kh - 1) * dilRow >= ih ||
        (ow - 1) * strideCol + (kw - 1) * dilCol >= iw)
      return failure();

    // Every value these pools read comes off a convolution with a relu, so it
    // is in [0,127] -- and then bit 7 is always clear, `(a|HI) - b` cannot
    // borrow out of its byte, and the whole sign dance goes away. Measured on
    // the board at GoogLeNet's shape: **-33.7%**, with the shorter formula
    // checked exhaustively over all 16,384 byte pairs it claims to cover.
    //
    // The accumulator starts from what the output buffer holds, and a max-pool
    // is filled with the i8 minimum. With a non-negative input the true maximum
    // is non-negative too, so that fill can be a zero -- and then the
    // accumulator is in range as well.
    bool unsignedSafe = false;
    if (nonNegativeI8(src, 0)) {
      if (linalg::FillOp fill = fillOf(out, pool)) {
        IntegerAttr cst;
        if (matchPattern(fill.getInputs()[0], m_Constant(&cst))) {
          if (cst.getValue().isNonNegative()) {
            unsignedSafe = true;
          } else {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(fill);
            Value zero = rewriter.create<arith::ConstantOp>(
                fill.getLoc(), rewriter.getI8IntegerAttr(0));
            rewriter.modifyOpInPlace(fill, [&] { fill->setOperand(0, zero); });
            unsignedSafe = true;
          }
        }
      }
    }

    Location loc = pool.getLoc();
    Type i64 = rewriter.getI64Type();
    Type idx = rewriter.getIndexType();

    auto words = [&](Value buffer, MemRefType ty) {
      int64_t total = ty.getNumElements();
      SmallVector<ReassociationIndices> group{{0, 1, 2, 3}};
      Value flat = rewriter.create<memref::CollapseShapeOp>(loc, buffer, group);
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      return rewriter.create<memref::ViewOp>(
          loc, MemRefType::get({total / kLanes}, i64), flat, zero, ValueRange{});
    };
    Value srcWords = words(srcBand->buffer, srcBand->type),
          outWords = words(outBand->buffer, outBand->type);

    auto cst = [&](int64_t v) {
      return rewriter.create<arith::ConstantIndexOp>(loc, v).getResult();
    };
    auto word = [&](uint64_t v) {
      return rewriter
          .create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(
                                              i64, (int64_t)v))
          .getResult();
    };
    Value hi = word(kHigh), low = word(kLow), mask = word(0xFF), seven = word(7);
    Value ones = word(~0ull);

    Value c0 = cst(0), c1 = cst(1);
    Value groups = cst(channels / kLanes);

    // In-word strides, in i64 units -- taken from the buffer the band views,
    // not from the band, because that is what the rows step over.
    int64_t srcRow = srcBand->type.getDimSize(2) * channels / kLanes;
    int64_t srcPlane = srcBand->type.getDimSize(1) * srcRow;
    int64_t outRow = outBand->type.getDimSize(2) * channels / kLanes;
    int64_t outPlane = outBand->type.getDimSize(1) * outRow;

    auto batch = rewriter.create<scf::ForOp>(loc, c0, cst(n), c1);
    rewriter.setInsertionPointToStart(batch.getBody());
    auto row = rewriter.create<scf::ForOp>(loc, c0, cst(oh), c1);
    rewriter.setInsertionPointToStart(row.getBody());
    auto col = rewriter.create<scf::ForOp>(loc, c0, cst(ow), c1);
    rewriter.setInsertionPointToStart(col.getBody());
    auto grp = rewriter.create<scf::ForOp>(loc, c0, groups, c1);
    rewriter.setInsertionPointToStart(grp.getBody());

    Value bi = batch.getInductionVar(), ri = row.getInductionVar(),
          ci = col.getInductionVar(), gi = grp.getInductionVar();

    auto mul = [&](Value v, int64_t k) {
      return k == 1 ? v
                    : rewriter.create<arith::MulIOp>(loc, v, cst(k)).getResult();
    };
    auto add = [&](Value a, Value b) {
      return rewriter.create<arith::AddIOp>(loc, a, b).getResult();
    };

    Value outIdx = add(add(mul(bi, outPlane), mul(ri, outRow)),
                       add(mul(ci, channels / kLanes), gi));
    if (outBand->baseWord)
      outIdx = add(outIdx, cst(outBand->baseWord));
    Value start = rewriter.create<memref::LoadOp>(loc, outWords, outIdx);
    Value acc = unsignedSafe
                    ? start
                    : rewriter.create<arith::XOrIOp>(loc, start, hi).getResult();

    // `scf.for` built with iteration arguments and no body builder has **no**
    // terminator -- MLIR leaves it to the caller, who is the only one who knows
    // what to yield.
    auto kRow = rewriter.create<scf::ForOp>(loc, c0, cst(kh), c1, ValueRange{acc});
    rewriter.setInsertionPointToStart(kRow.getBody());
    auto kCol = rewriter.create<scf::ForOp>(loc, c0, cst(kw), c1,
                                            ValueRange{kRow.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(kCol.getBody());

    Value srcRowIdx =
        add(mul(ri, strideRow), mul(kRow.getInductionVar(), dilRow));
    Value srcColIdx =
        add(mul(ci, strideCol), mul(kCol.getInductionVar(), dilCol));
    Value inIdx = add(add(mul(bi, srcPlane), mul(srcRowIdx, srcRow)),
                      add(mul(srcColIdx, channels / kLanes), gi));
    if (srcBand->baseWord)
      inIdx = add(inIdx, cst(srcBand->baseWord));
    Value raw = rewriter.create<memref::LoadOp>(loc, srcWords, inIdx);
    Value b = unsignedSafe
                  ? raw
                  : rewriter.create<arith::XOrIOp>(loc, raw, hi).getResult();
    Value a = kCol.getRegionIterArg(0);

    Value ge;
    if (unsignedSafe) {
      // Bit 7 of every byte is clear, so `0x80 + a_i - b_i` is in [1,255] and
      // cannot borrow out of its byte: bit 7 of the difference *is* `a_i >= b_i`.
      ge = rewriter.create<arith::SubIOp>(
          loc, rewriter.create<arith::OrIOp>(loc, a, hi), b);
    } else {
      // d = (a | HI) - (b & ~HI)
      Value notHi = rewriter.create<arith::XOrIOp>(loc, hi, ones);
      Value d = rewriter.create<arith::SubIOp>(
          loc, rewriter.create<arith::OrIOp>(loc, a, hi),
          rewriter.create<arith::AndIOp>(loc, b, notHi));
      // ge = HI & ((a & ~b) | (~(a ^ b) & d))
      Value notB = rewriter.create<arith::XOrIOp>(loc, b, ones);
      Value axb0 = rewriter.create<arith::XOrIOp>(loc, a, b);
      Value same = rewriter.create<arith::XOrIOp>(loc, axb0, ones);
      ge = rewriter.create<arith::AndIOp>(
          loc, hi,
          rewriter.create<arith::OrIOp>(
              loc, rewriter.create<arith::AndIOp>(loc, a, notB),
              rewriter.create<arith::AndIOp>(loc, same, d)));
    }
    Value axb = rewriter.create<arith::XOrIOp>(loc, a, b);
    // m = ((ge >> 7) & LOW) * 0xFF
    Value m = rewriter.create<arith::MulIOp>(
        loc,
        rewriter.create<arith::AndIOp>(
            loc, rewriter.create<arith::ShRUIOp>(loc, ge, seven), low),
        mask);
    // max = b ^ ((a ^ b) & m)
    Value next = rewriter.create<arith::XOrIOp>(
        loc, b, rewriter.create<arith::AndIOp>(loc, axb, m));
    rewriter.create<scf::YieldOp>(loc, ValueRange{next});
    rewriter.setInsertionPointToEnd(kRow.getBody());
    rewriter.create<scf::YieldOp>(loc, ValueRange{kCol.getResult(0)});

    rewriter.setInsertionPoint(grp.getBody()->getTerminator());
    Value done = unsignedSafe
                     ? kRow.getResult(0)
                     : rewriter.create<arith::XOrIOp>(loc, kRow.getResult(0), hi)
                           .getResult();
    rewriter.create<memref::StoreOp>(loc, done, outWords, outIdx);

    rewriter.eraseOp(pool);
    return success();
  }
};

class PackInt8MaxPool_Pass
    : public impl::PackInt8MaxPoolBase<PackInt8MaxPool_Pass> {
public:
  using impl::PackInt8MaxPoolBase<PackInt8MaxPool_Pass>::PackInt8MaxPoolBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PackInt8MaxPool>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
