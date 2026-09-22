//===- CombineConstantScalesPass.cpp -----------------------------*- C++ -*-===//
//
// Two constant coefficients on the dependency chain are one.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_COMBINECONSTANTSCALES
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

static bool floatConst(Value v, APFloat &out) {
  FloatAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return false;
  out = attr.getValue();
  return true;
}

/// One step of `a*x + b`, or nothing. `var` comes back as the operand the step
/// is applied to; `a` and `b` are multiplied into the running pair by
/// `applyStep`.
static bool isStep(Operation *op, Value *var = nullptr) {
  auto pick = [&](Value x) {
    if (var)
      *var = x;
    return true;
  };
  APFloat k(0.0f);
  if (auto o = llvm::dyn_cast<arith::NegFOp>(op))
    return pick(o.getOperand());
  if (llvm::isa<arith::MulFOp, arith::AddFOp>(op)) {
    if (floatConst(op->getOperand(1), k))
      return pick(op->getOperand(0));
    if (floatConst(op->getOperand(0), k))
      return pick(op->getOperand(1));
    return false;
  }
  if (llvm::isa<arith::DivFOp, arith::SubFOp>(op)) {
    if (floatConst(op->getOperand(1), k))
      return pick(op->getOperand(0));
    // `c / x` is not affine; `c - x` is.
    if (llvm::isa<arith::SubFOp>(op) && floatConst(op->getOperand(0), k))
      return pick(op->getOperand(1));
    return false;
  }
  if (auto o = llvm::dyn_cast<math::FmaOp>(op)) {
    if (floatConst(o.getOperand(1), k) && floatConst(o.getOperand(2), k))
      return pick(o.getOperand(0));
    if (floatConst(o.getOperand(0), k) && floatConst(o.getOperand(2), k))
      return pick(o.getOperand(1));
    return false;
  }
  return false;
}

/// Fold one step into the running `a*x + b`. The coefficients are computed in
/// the *result's* format, which is the one the target would have used.
static bool applyStep(Operation *op, APFloat &a, APFloat &b) {
  auto rm = APFloat::rmNearestTiesToEven;
  APFloat k(a.getSemantics()), k2(a.getSemantics());
  auto cst = [&](unsigned i, APFloat &out) {
    FloatAttr attr;
    if (!matchPattern(op->getOperand(i), m_Constant(&attr)))
      return false;
    out = attr.getValue();
    bool lost;
    return out.convert(a.getSemantics(), rm, &lost) == APFloat::opOK;
  };
  if (llvm::isa<arith::NegFOp>(op)) {
    a.changeSign();
    b.changeSign();
    return true;
  }
  if (llvm::isa<arith::MulFOp>(op)) {
    if (!cst(1, k) && !cst(0, k))
      return false;
    a.multiply(k, rm);
    b.multiply(k, rm);
    return true;
  }
  if (llvm::isa<arith::DivFOp>(op)) {
    if (!cst(1, k))
      return false;
    a.divide(k, rm);
    b.divide(k, rm);
    return true;
  }
  if (llvm::isa<arith::AddFOp>(op)) {
    if (!cst(1, k) && !cst(0, k))
      return false;
    b.add(k, rm);
    return true;
  }
  if (llvm::isa<arith::SubFOp>(op)) {
    if (cst(1, k)) { // x - c
      b.subtract(k, rm);
      return true;
    }
    if (!cst(0, k)) // c - x
      return false;
    a.changeSign();
    b.changeSign();
    b.add(k, rm);
    return true;
  }
  if (llvm::isa<math::FmaOp>(op)) {
    // a*x + b, then * k + k2
    if (!cst(2, k2))
      return false;
    if (!cst(1, k) && !cst(0, k))
      return false;
    a.multiply(k, rm);
    b.multiply(k, rm);
    b.add(k2, rm);
    return true;
  }
  return false;
}

/// The licence. A few ulps cannot move an answer that reaches nothing but a
/// conversion to an integer -- the same one `--reciprocal-for-division` and
/// `--combine-channel-affine` take -- and every operation allowed on the way is
/// monotone, so the rounding boundary is all that shifts.
static bool reachesIntegerConversion(Value v) {
  SmallVector<Value> work{v};
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    if (cur.use_empty())
      return false;
    for (Operation *user : cur.getUsers()) {
      if (llvm::isa<arith::FPToSIOp, arith::FPToUIOp>(user))
        continue;
      if (llvm::isa<math::RoundEvenOp, math::RoundOp, arith::MaxNumFOp,
                    arith::MinNumFOp, arith::MaximumFOp, arith::MinimumFOp>(
              user)) {
        work.push_back(user->getResult(0));
        continue;
      }
      return false;
    }
  }
  return true;
}

/// A quantization tail carries its constants one operation at a time:
///
/// ```
///   fcvt.s.w  fa3, a1              # the accumulator
///   fmadd.s   fa3, fa3, fa5, fs1   # * input scale + bias
///   fmul.s    fa3, fa3, fa4        # / output scale
///   fcvt.w.s  s0, fa3, rne
/// ```
///
/// Both coefficients are known at compile time and both sit **on the dependency
/// chain**, which is what an in-order single-issue core actually pays
/// ([[gemmlir-the-chain-not-the-count]]). `(a*x + b)*k` is `a*k*x + b*k`: one
/// operation, and the two constants folded once by the compiler.
///
/// The reassociation is licensed the way the reciprocal is -- the value reaches
/// nothing but a conversion to an integer -- and refused outright when folding
/// the constants together stops being finite, which is the one way a few ulps
/// could turn into a different answer.
class CombineConstantScales : public RewritePattern {
public:
  CombineConstantScales(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    if (!isStep(op))
      return failure();
    Value result = op->getResult(0);
    Type ty = result.getType();
    if (!ty.isF32() && !ty.isF64())
      return failure();

    // Anchor on the last step: if the one above continues the chain it will do
    // the whole of it.
    for (Operation *user : result.getUsers()) {
      Value var;
      if (result.hasOneUse() && isStep(user, &var) && var == result)
        return failure();
    }

    SmallVector<Operation *> chain;
    Operation *cursor = op;
    Value root;
    while (true) {
      chain.push_back(cursor);
      Value var;
      isStep(cursor, &var);
      root = var;
      Operation *def = var.getDefiningOp();
      if (!def || !var.hasOneUse() || def->getResult(0).getType() != ty ||
          !isStep(def))
        break;
      cursor = def;
    }
    if (chain.size() < 2)
      return failure();
    if (!reachesIntegerConversion(result))
      return failure();

    const llvm::fltSemantics &sem =
        ty.isF64() ? APFloat::IEEEdouble() : APFloat::IEEEsingle();
    APFloat a(sem, 1), b(sem, 0);
    for (Operation *step : llvm::reverse(chain))
      if (!applyStep(step, a, b))
        return failure();
    // Folding two constants together is the one way this could change an answer
    // by more than a rounding boundary.
    if (!a.isFinite() || !b.isFinite() || a.isZero())
      return failure();

    Location loc = op->getLoc();
    Value av = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(ty, a));
    Value out;
    if (b.isZero()) {
      out = rewriter.create<arith::MulFOp>(loc, root, av);
    } else {
      Value bv = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(ty, b));
      out = rewriter.create<math::FmaOp>(loc, root, av, bv);
    }
    rewriter.replaceOp(op, out);
    return success();
  }
};

class CombineConstantScales_Pass
    : public impl::CombineConstantScalesBase<CombineConstantScales_Pass> {
public:
  using impl::CombineConstantScalesBase<
      CombineConstantScales_Pass>::CombineConstantScalesBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<CombineConstantScales>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
