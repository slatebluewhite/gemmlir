//===- CombineChannelAffinePass.cpp ------------------------------*- C++ -*-===//
//
// Two per-channel numbers instead of three operations an element.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_COMBINECHANNELAFFINE
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// True when `v` is only ever read by float arithmetic that ends in a
/// conversion to an integer -- which is what a quantization tail is.
///
/// This is the same condition `--hoist-invariant-reciprocal` turns on, and for
/// the same reason: the rewrite is exact in real arithmetic and differs by
/// rounding in float, so it has to reach a place where a few ulps cannot change
/// the answer.
static bool reachesOnlyIntegerConversion(Value v, unsigned depth = 0) {
  if (depth > 16)
    return false;
  for (Operation *user : v.getUsers()) {
    if (llvm::isa<arith::FPToSIOp, arith::FPToUIOp>(user))
      continue;
    if (!llvm::isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                   arith::NegFOp, arith::MaximumFOp, arith::MaxNumFOp,
                   arith::MinimumFOp, arith::MinNumFOp, arith::SelectOp,
                   arith::CmpFOp, math::RoundEvenOp>(user))
      return false;
    if (llvm::isa<arith::CmpFOp>(user))
      continue; // a comparison yields an i1; it cannot carry the value on
    if (user->getNumResults() != 1 ||
        !reachesOnlyIntegerConversion(user->getResult(0), depth + 1))
      return false;
  }
  return true;
}

static bool isZeroConst(Value v) {
  llvm::APFloat f(0.0f);
  return matchPattern(v, m_ConstantFloat(&f)) && f.isZero() && !f.isNegative();
}

static bool isPositiveConst(Value v) {
  llvm::APFloat f(0.0f);
  return matchPattern(v, m_ConstantFloat(&f)) && !f.isNaN() && !f.isNegative() &&
         !f.isZero();
}

/// `max(y, 0)` however it is written, or nothing.
static Value reluInput(Operation *op) {
  if (auto sel = llvm::dyn_cast<arith::SelectOp>(op)) {
    auto cmp = sel.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!cmp)
      return nullptr;
    bool greater = cmp.getPredicate() == arith::CmpFPredicate::UGT ||
                   cmp.getPredicate() == arith::CmpFPredicate::OGT;
    if (!greater || cmp.getLhs() != sel.getTrueValue() ||
        cmp.getRhs() != sel.getFalseValue() || !isZeroConst(sel.getFalseValue()))
      return nullptr;
    return sel.getTrueValue();
  }
  if (llvm::isa<arith::MaximumFOp, arith::MaxNumFOp>(op)) {
    if (isZeroConst(op->getOperand(1)))
      return op->getOperand(0);
    if (isZeroConst(op->getOperand(0)))
      return op->getOperand(1);
  }
  return nullptr;
}

/// A batch norm that could not fold into anybody's weights spends three
/// operations an element on two numbers that only change per channel:
///
/// ```
///   fsub.s  fa3, fa3, fa2        # x - mean[c]
///   fmadd.s fa3, fa3, fa1, fa5   # * rsqrt[c] + beta
///   ...relu...
///   fmul.s  fa3, fa3, fa4        # / scale
///   fcvt.w.s a1, fa3, rne
/// ```
///
/// DenseNet-121 is 593 of them -- every dense layer renormalizes the whole
/// concatenated stack, and the parameters belong to the *consumer*, so
/// `--fold-batch-norm` has no weights to put them in. `fsub.s` and `fmul.s`
/// alone are **11.6% of the model** by program-counter sampling, and they are
/// two of the five steps on the dependency chain, which is what an in-order core
/// actually pays ([[gemmlir-the-chain-not-the-count]]).
///
/// `((x - m) * r + b) / s` is `x * (r/s) + (b - m*r)/s`: **one** multiply-add on
/// two per-channel numbers. The scale moves inside the relu, which is sound
/// because `max(y, 0)/s == max(y/s, 0)` for a positive `s` -- and on a NaN or a
/// negative zero both forms do the same thing, because the comparison is
/// unordered either way and `-0/s` is `-0`.
///
/// The two forms differ by rounding, so the rewrite is confined to a value that
/// **reaches a conversion to an integer and nothing else** -- the same condition
/// and the same argument as `--hoist-invariant-reciprocal`.
class CombineChannelAffine : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.hasPureBufferSemantics() ||
        generic.getRegion().getBlocks().size() != 1)
      return failure();
    Block &body = generic.getRegion().front();
    unsigned numLoops = generic.getNumLoops();
    if (generic.getNumDpsInputs() < 3)
      return failure();

    // The anchor is the scale below the activation.
    for (Operation &op : body) {
      Value scale;
      bool divide = false;
      if (auto d = llvm::dyn_cast<arith::DivFOp>(&op)) {
        scale = d.getRhs();
        divide = true;
      } else if (auto m = llvm::dyn_cast<arith::MulFOp>(&op)) {
        scale = m.getRhs();
      } else {
        continue;
      }
      if (!isPositiveConst(scale))
        continue;
      Operation *reluOp = op.getOperand(0).getDefiningOp();
      if (!reluOp || !op.getOperand(0).hasOneUse())
        continue;
      Value y = reluInput(reluOp);
      // A relu written as a compare and a select reads the value **twice**, so
      // `hasOneUse` is the wrong test -- it refused all 593 of DenseNet's.
      // What matters is that nothing outside the activation reads it.
      if (!y || llvm::any_of(y.getUsers(), [&](Operation *u) {
            return u != reluOp && u != reluOp->getOperand(0).getDefiningOp();
          }))
        continue;
      auto add = y.getDefiningOp<arith::AddFOp>();
      if (!add)
        continue;
      Value beta = add.getRhs();
      llvm::APFloat betaValue(0.0f);
      if (!matchPattern(beta, m_ConstantFloat(&betaValue)) ||
          betaValue.isNaN() || !add.getLhs().hasOneUse())
        continue;
      auto mul = add.getLhs().getDefiningOp<arith::MulFOp>();
      if (!mul)
        continue;
      auto rArg = llvm::dyn_cast<BlockArgument>(mul.getRhs());
      if (!rArg || !mul.getLhs().hasOneUse())
        continue;
      auto sub = mul.getLhs().getDefiningOp<arith::SubFOp>();
      if (!sub || sub.getLhs() != body.getArgument(0))
        continue;
      auto mArg = llvm::dyn_cast<BlockArgument>(sub.getRhs());
      if (!mArg || mArg == rArg || !mArg.hasOneUse() || !rArg.hasOneUse())
        continue;
      if (!reachesOnlyIntegerConversion(op.getResult(0)))
        continue;

      // Both per-channel operands have to be read the same way, or one loop
      // cannot produce both coefficients.
      OpOperand *mOperand = &generic->getOpOperand(mArg.getArgNumber());
      OpOperand *rOperand = &generic->getOpOperand(rArg.getArgNumber());
      AffineMap mMap = generic.getMatchingIndexingMap(mOperand);
      AffineMap rMap = generic.getMatchingIndexingMap(rOperand);
      if (mMap != rMap || !rMap.isProjectedPermutation() ||
          rMap.getNumResults() >= numLoops)
        continue;
      auto memTy = llvm::dyn_cast<MemRefType>(rOperand->get().getType());
      auto other = llvm::dyn_cast<MemRefType>(mOperand->get().getType());
      if (!memTy || !other || memTy != other || !memTy.hasStaticShape() ||
          !llvm::isa<FloatType>(memTy.getElementType()))
        continue;

      rewrite(generic, mOperand, rOperand, mArg, rArg, memTy, scale, beta,
              divide, &op, sub, mul, add, rewriter);
      return success();
    }
    return failure();
  }

private:
  void rewrite(linalg::GenericOp generic, OpOperand *mOperand,
               OpOperand *rOperand, BlockArgument mArg, BlockArgument rArg,
               MemRefType memTy, Value scale, Value beta, bool divide,
               Operation *scaleOp, arith::SubFOp sub, arith::MulFOp mul,
               arith::AddFOp add, PatternRewriter &rewriter) const {
    Location loc = generic.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(generic);

    Value bufA = rewriter.create<memref::AllocOp>(
        loc, memTy, rewriter.getI64IntegerAttr(64));
    Value bufB = rewriter.create<memref::AllocOp>(
        loc, memTy, rewriter.getI64IntegerAttr(64));

    unsigned rank = memTy.getRank();
    SmallVector<AffineMap> maps(
        4, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    Value mBuf = mOperand->get(), rBuf = rOperand->get();
    rewriter.create<linalg::GenericOp>(
        loc, TypeRange{}, ValueRange{mBuf, rBuf}, ValueRange{bufA, bufB}, maps,
        iters, [&](OpBuilder &b, Location nested, ValueRange args) {
          Value m = args[0], r = args[1];
          Value s = b.clone(*scale.getDefiningOp())->getResult(0);
          Value bt = b.clone(*beta.getDefiningOp())->getResult(0);
          auto apply = [&](Value v) -> Value {
            return divide ? (Value)b.create<arith::DivFOp>(nested, v, s)
                          : (Value)b.create<arith::MulFOp>(nested, v, s);
          };
          Value a = apply(r);
          Value mr = b.create<arith::MulFOp>(nested, m, r);
          Value num = b.create<arith::SubFOp>(nested, bt, mr);
          b.create<linalg::YieldOp>(nested, ValueRange{a, apply(num)});
        });

    // `mean` becomes `B` and `rsqrt` becomes `A`: the block arguments keep
    // their types and their maps, so the operation does not have to be rebuilt.
    rewriter.modifyOpInPlace(generic, [&] {
      mOperand->set(bufB);
      rOperand->set(bufA);
    });

    rewriter.setInsertionPoint(sub);
    Value prod = rewriter.create<arith::MulFOp>(
        loc, generic.getRegion().front().getArgument(0), rArg);
    Value sum = rewriter.create<arith::AddFOp>(loc, prod, mArg);
    rewriter.replaceAllUsesWith(add.getResult(), sum);
    rewriter.replaceAllUsesWith(scaleOp->getResult(0), scaleOp->getOperand(0));
    rewriter.eraseOp(scaleOp);
    rewriter.eraseOp(add);
    rewriter.eraseOp(mul);
    rewriter.eraseOp(sub);
  }
};

class CombineChannelAffine_Pass
    : public impl::CombineChannelAffineBase<CombineChannelAffine_Pass> {
public:
  using impl::CombineChannelAffineBase<
      CombineChannelAffine_Pass>::CombineChannelAffineBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, arith::ArithDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<CombineChannelAffine>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
