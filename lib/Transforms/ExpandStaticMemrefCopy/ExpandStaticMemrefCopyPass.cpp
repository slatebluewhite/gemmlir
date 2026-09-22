//===- ExpandStaticMemrefCopyPass.cpp ----------------------------*- C++ -*-===//
//
// A copy whose shape is known here does not need a runtime that reads it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Alignment.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_EXPANDSTATICMEMREFCOPY
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The widest load this core has.
constexpr int64_t kWord = 8;
/// A run of exactly one word is the one size LLVM takes apart: `vector<8xi8>`
/// comes out as eight `lbu` and eight `sb`, where `vector<16xi8>` and up are
/// split into whole `ld`/`sd` pairs. Measured on the RISC-V backend.
constexpr int64_t kLeastRunBytes = 16;
/// Past this the copy is left alone, and the runtime takes it.
///
/// Which is fine, and the check that it is fine was worth making. A
/// `memref.copy` whose two sides have different layouts -- every copy into the
/// middle of a padding, every copy into one branch's slice of a join -- lowers
/// to a call to `memrefCopy`, and **MLIR's** `memrefCopy` walks the index space
/// one element at a time calling `memcpy(dst, src, elemSize)` on each. That
/// would be worth removing. But the generated code does not reach MLIR's: this
/// project ships its own `memrefCopy` in `runtime/`, which copies a packed run
/// at a time. PC sampling puts it at **0.3%** of GoogLeNet.
///
/// Expanding those eighteen copies into loops of `memcpy` was built and
/// measured against it anyway: `googlenet` -0.2%, `squeezenet1_1` +1.1%, the
/// set **-0.0%**. Reverted. The fix was already there, one layer down.
constexpr int64_t kMostRunBytes = 64;
/// A run that is not a whole number of words cannot be moved as aligned
/// vectors, and the runtime copies it one byte at a time with the length in a
/// register -- 94 cycles for the nine bytes of an im2col pack over a
/// three-channel image. Unrolled here at constant offsets it is nine loads and
/// nine stores and no inner loop at all. Only while that stays short.
/// **Twenty-four, not sixteen.** DenseNet's stem packs im2col over a
/// three-channel image with a 7x7 kernel, which is a run of 7*3 = **21 bytes**
/// -- just over the old cap, so all 7168 of its runs went to the runtime
/// instead. PC sampling put that one copy at 3.0% of the model, and raising the
/// cap takes `densenet121` 721.22 -> **712.51** ms, -1.2%, byte for byte
/// against both references. Nothing else in the set has a run in 17..24.
///
/// Sixteen was never measured -- it was "only while that stays short", written
/// when the example to hand was nine bytes. A kernel size decides this number,
/// and 7x7 over three channels is an ordinary stem.
constexpr int64_t kMostUnrolledElements = 24;

/// The byte alignment a memref's base pointer is known to have, or 0 when
/// nothing here says. Only the *base*: the offset in the layout is checked
/// separately, because that is where a subview's or a cast's own displacement
/// ends up.
static int64_t baseAlignment(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return 0;
  if (auto alloc = llvm::dyn_cast<memref::AllocOp>(def))
    return alloc.getAlignment().value_or(0);
  if (auto alloca = llvm::dyn_cast<memref::AllocaOp>(def))
    return alloca.getAlignment().value_or(0);
  if (auto get = llvm::dyn_cast<memref::GetGlobalOp>(def)) {
    auto module = get->getParentOfType<ModuleOp>();
    auto global = module.lookupSymbol<memref::GlobalOp>(get.getNameAttr());
    return global ? global.getAlignment().value_or(0) : 0;
  }
  // `memref.view`'s shift is in bytes and does *not* appear in the result's
  // layout, so it has to be folded in here.
  if (auto view = llvm::dyn_cast<memref::ViewOp>(def)) {
    llvm::APInt shift;
    if (!matchPattern(view.getByteShift(), m_ConstantInt(&shift)))
      return 0;
    int64_t base = baseAlignment(view.getSource());
    int64_t bytes = shift.getSExtValue();
    if (bytes == 0)
      return base;
    return std::min(base, bytes & -bytes);
  }
  if (llvm::isa<memref::SubViewOp, memref::ReinterpretCastOp,
                memref::CollapseShapeOp, memref::ExpandShapeOp, memref::CastOp>(
          def))
    return baseAlignment(def->getOperand(0));
  return 0;
}

/// `memref.copy` between two views whose shapes and strides are all known here
/// becomes the loop nest the runtime would have walked, with the run moved as
/// one vector.
///
/// The runtime's copy reads a descriptor, works out how much of the shape is
/// packed on both sides, decides whether the run is word-aligned, and then runs
/// an odometer -- all of which is the same answer every time for a given call
/// site. Written out here the run length is a constant, so the loads and stores
/// are unrolled, and the index arithmetic is two `addi`. Measured on `gmid`'s
/// im2col pack, 768 runs of 24 bytes: **36 -> 19.7 cycles a run**.
///
/// What it will not do:
///
///   * a run of exactly one word, which LLVM takes apart into byte accesses;
///   * a run longer than 64 bytes, where the runtime's `memcpy` is better;
///   * anything it cannot prove eight-byte aligned -- both bases, both layout
///     offsets and every remaining stride. An unaligned `vector.load` is byte
///     accesses again, which is slower than the call it replaced.
class ExpandCopy : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const final {
    auto srcTy = llvm::dyn_cast<MemRefType>(copy.getSource().getType());
    auto dstTy = llvm::dyn_cast<MemRefType>(copy.getTarget().getType());
    if (!srcTy || !dstTy || !srcTy.hasStaticShape() || !dstTy.hasStaticShape() ||
        srcTy.getShape() != dstTy.getShape() ||
        srcTy.getElementType() != dstTy.getElementType())
      return failure();

    Type elem = srcTy.getElementType();
    if (!elem.isIntOrFloat())
      return failure();
    unsigned bits = elem.getIntOrFloatBitWidth();
    if (bits == 0 || bits % 8 != 0)
      return failure();
    int64_t elemBytes = bits / 8;

    SmallVector<int64_t> srcStrides, dstStrides;
    int64_t srcOffset = 0, dstOffset = 0;
    if (failed(srcTy.getStridesAndOffset(srcStrides, srcOffset)) ||
        failed(dstTy.getStridesAndOffset(dstStrides, dstOffset)) ||
        ShapedType::isDynamic(srcOffset) || ShapedType::isDynamic(dstOffset))
      return failure();
    for (int64_t s : srcStrides)
      if (ShapedType::isDynamic(s))
        return failure();
    for (int64_t s : dstStrides)
      if (ShapedType::isDynamic(s))
        return failure();

    // The longest suffix that is packed in both operands is the run.
    int64_t rank = srcTy.getRank();
    ArrayRef<int64_t> shape = srcTy.getShape();
    int64_t run = 1, suffix = 0;
    while (suffix < rank && srcStrides[rank - 1 - suffix] == run &&
           dstStrides[rank - 1 - suffix] == run) {
      run *= shape[rank - 1 - suffix];
      suffix++;
    }
    int64_t outer = rank - suffix;
    // Packed all the way through is one `memcpy` already, which the existing
    // lowering emits and does better.
    if (outer < 1)
      return failure();
    int64_t runBytes = run * elemBytes;
    if (runBytes > kMostRunBytes)
      return failure();

    // A whole number of words, aligned end to end, moves as one vector. A run
    // that is not moves as its elements, which needs nothing proved: a load of
    // the element type is aligned wherever the element is.
    bool wordwise = runBytes % kWord == 0;
    if (wordwise) {
      if (runBytes < kLeastRunBytes)
        return failure();
      auto startsAligned = [&](Value v, int64_t offset) {
        return baseAlignment(v) >= kWord && (offset * elemBytes) % kWord == 0;
      };
      if (!startsAligned(copy.getSource(), srcOffset) ||
          !startsAligned(copy.getTarget(), dstOffset))
        return failure();
      for (int64_t d = 0; d < outer; d++)
        if ((srcStrides[d] * elemBytes) % kWord != 0 ||
            (dstStrides[d] * elemBytes) % kWord != 0)
          return failure();
    } else if (run > kMostUnrolledElements) {
      return failure();
    }

    Location loc = copy.getLoc();
    Value src = copy.getSource(), dst = copy.getTarget();
    // Make the run one dimension, so the vector reads the innermost axis and
    // nothing has to reason about reading across a boundary.
    if (suffix > 1) {
      SmallVector<ReassociationIndices> groups;
      for (int64_t d = 0; d < outer; d++)
        groups.push_back({d});
      ReassociationIndices tail;
      for (int64_t d = outer; d < rank; d++)
        tail.push_back(d);
      groups.push_back(tail);
      src = rewriter.create<memref::CollapseShapeOp>(loc, src, groups);
      dst = rewriter.create<memref::CollapseShapeOp>(loc, dst, groups);
    }

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> lbs(outer, zero), steps(outer, one), ubs;
    for (int64_t d = 0; d < outer; d++)
      ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, shape[d]));

    auto vecTy = VectorType::get({run}, elem);
    llvm::MaybeAlign align(kWord);
    scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location l, ValueRange ivs) {
          SmallVector<Value> at(ivs.begin(), ivs.end());
          at.push_back(zero);
          if (wordwise) {
            Value v = b.create<vector::LoadOp>(l, vecTy, src, at,
                                               /*nontemporal=*/false, align);
            b.create<vector::StoreOp>(l, v, dst, at, /*nontemporal=*/false,
                                      align);
            return;
          }
          for (int64_t e = 0; e < run; e++) {
            at.back() = b.create<arith::ConstantIndexOp>(l, e);
            Value v = b.create<memref::LoadOp>(l, src, at);
            b.create<memref::StoreOp>(l, v, dst, at);
          }
        });
    rewriter.eraseOp(copy);
    return success();
  }
};

class ExpandStaticMemrefCopy
    : public impl::ExpandStaticMemrefCopyBase<ExpandStaticMemrefCopy> {
public:
  using impl::ExpandStaticMemrefCopyBase<
      ExpandStaticMemrefCopy>::ExpandStaticMemrefCopyBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandCopy>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
