//===- HoistInvariantReciprocalPass.cpp --------------------------*- C++ -*-===//
//
// Dividing by something the inner loop never changes is multiplying by a
// reciprocal computed once.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_HOISTINVARIANTRECIPROCAL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// True when `v` is only ever read by float arithmetic that ends in a
/// conversion to an integer -- which is what a quantization tail is.
///
/// This is the pass's soundness condition, not a heuristic. See the comment on
/// `DivideByRowInvariant`.
static bool reachesOnlyIntegerConversion(Value v) {
  SmallVector<Value> work{v};
  SmallPtrSet<Operation *, 8> seen;
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    for (Operation *user : cur.getUsers()) {
      if (!seen.insert(user).second)
        continue;
      if (llvm::isa<arith::FPToSIOp, arith::FPToUIOp>(user))
        continue;
      // Float arithmetic on the way there is fine; anything that stores the
      // value, yields it or widens it out of the block is not.
      if (!llvm::isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                     arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
                     arith::MaxNumFOp, arith::MinNumFOp, math::RoundEvenOp,
                     math::RoundOp, math::FloorOp, math::CeilOp,
                     math::AbsFOp>(user))
        return false;
      if (user->getNumResults() != 1)
        return false;
      work.push_back(user->getResult(0));
    }
  }
  return true;
}

/// `x / b` becomes `x * (1/b)` when `b` is read through a map that drops at
/// least one of the nest's loops -- so one reciprocal serves a whole row.
///
/// **Why.** `fdiv.s` on this in-order core is about 22 cycles and does not
/// pipeline. A softmax divides every element of its score matrix by that row's
/// sum of exponentials: 4096 divisions where 64 would do. Measured on the board
/// on `sa`, one head of self-attention at 64 tokens and 64 channels:
/// **20.81 -> 17.84 ms**, with the relative L2 against PyTorch unchanged at
/// 0.0183.
///
/// **What it costs.** The same trade `DivideByConstant` makes -- the reciprocal
/// is rounded once, so the product can differ from the quotient by an ulp.
///
/// **The part a constant divisor did not have.** `DivideByConstant` can look at
/// `1/c` and refuse when it is denormal or infinite. Here `b` is a runtime
/// value, so two cases have to be argued rather than checked:
///
///  * `b` denormal or zero makes `1/b` infinite, and `x * inf` is infinite
///    where `x / b` was merely large. The reciprocal loop clamps to
///    `copysign(FLT_MAX, 1/b)`, which keeps it finite.
///  * `b` above 2^126 makes `1/b` denormal and the product loses bits that the
///    quotient would have kept.
///
/// Both are why the rewrite is confined to a quotient that **reaches a
/// conversion to an integer and nothing else**. On those inputs the two forms
/// agree in the only place anyone looks: where the clamp bites, `x / b` was
/// already far outside the integer's range and the conversion was already
/// undefined; where `1/b` is denormal, `x / b` is itself so near zero that both
/// forms round to the same integer.
///
/// Buffer semantics only, and it must run **before** `--plan-static-buffers`,
/// which is what turns the reciprocal's buffer into a static one. Left to the
/// allocator it would be a malloc per inference -- the thing that pass exists
/// to remove.
class DivideByRowInvariant : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.hasPureBufferSemantics())
      return failure();
    if (generic.getRegion().getBlocks().size() != 1)
      return failure();
    Block &body = generic.getRegion().front();
    unsigned numLoops = generic.getNumLoops();

    for (OpOperand *operand : generic.getDpsInputOperands()) {
      auto memTy = llvm::dyn_cast<MemRefType>(operand->get().getType());
      if (!memTy || !memTy.hasStaticShape())
        continue;
      auto elemTy = llvm::dyn_cast<FloatType>(memTy.getElementType());
      if (!elemTy)
        continue;

      // Invariant along at least one loop, and read in the plain way -- a
      // projection, so every element of the reciprocal buffer stands for one
      // element of the divisor.
      AffineMap map = generic.getMatchingIndexingMap(operand);
      if (!map.isProjectedPermutation() || map.getNumResults() >= numLoops)
        continue;

      BlockArgument arg = body.getArgument(operand->getOperandNumber());
      if (arg.use_empty())
        continue;

      // Every use has to be a divisor; rewriting one of two roles would leave
      // the operand read twice for no gain.
      SmallVector<arith::DivFOp> divisions;
      bool onlyDivisor = true;
      for (OpOperand &use : arg.getUses()) {
        auto div = llvm::dyn_cast<arith::DivFOp>(use.getOwner());
        if (!div || use.getOperandNumber() != 1) {
          onlyDivisor = false;
          break;
        }
        divisions.push_back(div);
      }
      if (!onlyDivisor || divisions.empty())
        continue;

      if (!llvm::all_of(divisions, [](arith::DivFOp div) {
            return reachesOnlyIntegerConversion(div.getResult());
          }))
        continue;

      rewrite(generic, operand, arg, divisions, memTy, elemTy, rewriter);
      return success();
    }
    return failure();
  }

private:
  void rewrite(linalg::GenericOp generic, OpOperand *operand, BlockArgument arg,
               ArrayRef<arith::DivFOp> divisions, MemRefType memTy,
               FloatType elemTy, PatternRewriter &rewriter) const {
    Location loc = generic.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(generic);

    Value buffer = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(memTy.getShape(), elemTy),
        rewriter.getI64IntegerAttr(64));

    // r[i] = 1/b[i], kept finite so that a denormal `b` cannot turn a large
    // quotient into an infinite product.
    unsigned rank = memTy.getRank();
    SmallVector<AffineMap> maps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    rewriter.create<linalg::GenericOp>(
        loc, TypeRange{}, ValueRange{operand->get()}, ValueRange{buffer}, maps,
        iters,
        [&](OpBuilder &b, Location nested, ValueRange args) {
          Value one = b.create<arith::ConstantOp>(
              nested, elemTy, b.getFloatAttr(elemTy, 1.0));
          Value inv = b.create<arith::DivFOp>(nested, one, args[0]);
          llvm::APFloat biggest = llvm::APFloat::getLargest(
              elemTy.getFloatSemantics(), /*Negative=*/false);
          Value largest = b.create<arith::ConstantOp>(
              nested, elemTy, b.getFloatAttr(elemTy, biggest));
          Value magnitude = b.create<math::AbsFOp>(nested, inv);
          Value finite = b.create<arith::CmpFOp>(
              nested, arith::CmpFPredicate::OLE, magnitude, largest);
          Value clamped = b.create<math::CopySignOp>(nested, largest, inv);
          Value safe = b.create<arith::SelectOp>(nested, finite, inv, clamped);
          b.create<linalg::YieldOp>(nested, safe);
        });

    rewriter.modifyOpInPlace(generic, [&]() {
      generic->setOperand(operand->getOperandNumber(), buffer);
    });
    for (arith::DivFOp div : divisions) {
      rewriter.setInsertionPoint(div);
      rewriter.replaceOpWithNewOp<arith::MulFOp>(div, div.getLhs(), arg);
    }

    rewriter.setInsertionPointAfter(generic);
    rewriter.create<memref::DeallocOp>(loc, buffer);
  }
};

/// A per-channel `rsqrt` computed once per channel rather than per element.
///
/// A batch norm is `(x - mean[c]) * rsqrt(var[c] + eps)`, and the `rsqrt` half
/// depends only on the channel. `--fold-batch-norm` takes the whole thing into
/// the weights of the contraction above it, and where there is one this never
/// sees it; where there is **not** one -- a DenseNet layer's batch norm sits on
/// a join of everything the block has produced -- the region stays, and the
/// `rsqrt` runs on every element. On `densenet121` that is **8.0 million** of
/// them across 593 regions, where 593 x 64 would do.
///
/// The rewrite is the same shape as the reciprocal above: evaluate the
/// invariant part into a buffer of the operand's own size, then read it. What
/// makes it sound is narrower, though, and does not need that pass's argument
/// about denormals: `rsqrt(b + c)` is a **function of `b` alone**, so the
/// buffer holds exactly what the body would have computed, bit for bit. There
/// is no reassociation and no reciprocal, so the result is the same float.
///
/// It runs in `MID`, after every fold has been decided -- rewriting a batch
/// norm earlier is what breaks `matchRequantize`; see the note in
/// `FoldBatchNormPass.cpp`.
class RsqrtOfInvariant : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.hasPureBufferSemantics())
      return failure();
    if (generic.getRegion().getBlocks().size() != 1)
      return failure();
    Block &body = generic.getRegion().front();
    unsigned numLoops = generic.getNumLoops();

    for (OpOperand *operand : generic.getDpsInputOperands()) {
      auto memTy = llvm::dyn_cast<MemRefType>(operand->get().getType());
      if (!memTy || !memTy.hasStaticShape())
        continue;
      auto elemTy = llvm::dyn_cast<FloatType>(memTy.getElementType());
      if (!elemTy)
        continue;
      AffineMap map = generic.getMatchingIndexingMap(operand);
      if (!map.isProjectedPermutation() || map.getNumResults() >= numLoops)
        continue;

      BlockArgument arg = body.getArgument(operand->getOperandNumber());
      if (arg.use_empty())
        continue;

      // Every use is the `rsqrt` of this operand plus something the loop does
      // not change. Rewriting one of two roles would leave the operand read
      // twice for no gain.
      SmallVector<math::RsqrtOp> roots;
      bool onlyRoot = true;
      for (OpOperand &use : arg.getUses()) {
        Operation *user = use.getOwner();
        math::RsqrtOp root;
        if (auto add = llvm::dyn_cast<arith::AddFOp>(user)) {
          Value other = use.getOperandNumber() == 0 ? add.getRhs() : add.getLhs();
          if (!isInvariant(other, body) || !add.getResult().hasOneUse()) {
            onlyRoot = false;
            break;
          }
          root = llvm::dyn_cast<math::RsqrtOp>(*add.getResult().getUsers().begin());
        } else {
          root = llvm::dyn_cast<math::RsqrtOp>(user);
        }
        if (!root) {
          onlyRoot = false;
          break;
        }
        roots.push_back(root);
      }
      if (!onlyRoot || roots.empty())
        continue;

      rewrite(generic, operand, arg, roots, memTy, elemTy, rewriter);
      return success();
    }
    return failure();
  }

private:
  /// A value the loop does not change.
  ///
  /// Defined outside the body is the easy case. The one that matters here is
  /// **inside** it: a batch norm's epsilon reaches the region as an f64
  /// `arith.constant` outside and an `arith.truncf` in, so the addend is a
  /// body operation that reads nothing per-element. Refusing those meant the
  /// pattern never fired on any of DenseNet's 593 batch norms.
  static bool isInvariant(Value v, Block &body, unsigned depth = 0) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return false; // a block argument is per-element by construction
    if (def->getBlock() != &body)
      return true;
    if (depth > 8 || !isMemoryEffectFree(def))
      return false;
    return llvm::all_of(def->getOperands(), [&](Value operand) {
      return isInvariant(operand, body, depth + 1);
    });
  }

  void rewrite(linalg::GenericOp generic, OpOperand *operand, BlockArgument arg,
               ArrayRef<math::RsqrtOp> roots, MemRefType memTy,
               FloatType elemTy, PatternRewriter &rewriter) const {
    Location loc = generic.getLoc();
    Block &body = generic.getRegion().front();
    // One root's chain is the recipe; the others are the same chain on the same
    // operand, so replaying it once is enough.
    math::RsqrtOp first = roots.front();
    Operation *addOp = first.getOperand().getDefiningOp();
    auto add = llvm::dyn_cast_or_null<arith::AddFOp>(addOp);
    Value other;
    if (add)
      other = add.getLhs() == arg ? add.getRhs() : add.getLhs();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(generic);
    Value buffer = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(memTy.getShape(), elemTy),
        rewriter.getI64IntegerAttr(64));

    unsigned rank = memTy.getRank();
    SmallVector<AffineMap> maps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    rewriter.create<linalg::GenericOp>(
        loc, TypeRange{}, ValueRange{operand->get()}, ValueRange{buffer}, maps,
        iters, [&](OpBuilder &b, Location nested, ValueRange args) {
          Value v = args[0];
          if (other) {
            // The addend may be a computation *inside* the old body -- an
            // epsilon is an `arith.truncf` of an f64 constant. It cannot be
            // referenced from out here, so it is cloned in.
            IRMapping into;
            Value addend = other;
            if (Operation *def = other.getDefiningOp())
              if (def->getBlock() == &body)
                addend = b.clone(*def, into)->getResult(0);
            v = b.create<arith::AddFOp>(nested, v, addend);
          }
          b.create<linalg::YieldOp>(nested,
                                    b.create<math::RsqrtOp>(nested, v).getResult());
        });

    rewriter.modifyOpInPlace(generic, [&]() {
      generic->setOperand(operand->getOperandNumber(), buffer);
    });
    for (math::RsqrtOp root : roots)
      rewriter.replaceOp(root, arg);

    rewriter.setInsertionPointAfter(generic);
    rewriter.create<memref::DeallocOp>(loc, buffer);
  }
};

class HoistInvariantReciprocal
    : public impl::HoistInvariantReciprocalBase<HoistInvariantReciprocal> {
public:
  using impl::HoistInvariantReciprocalBase<
      HoistInvariantReciprocal>::HoistInvariantReciprocalBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<DivideByRowInvariant, RsqrtOfInvariant>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
