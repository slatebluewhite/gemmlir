//===- DropClampBelowReluPass.cpp --------------------------------*- C++ -*-===//
//
// A quantizer below an activation does not need its lower clamp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_DROPCLAMPBELOWRELU
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

static bool isNonNegativeConst(Value v) {
  llvm::APFloat f(0.0f);
  return matchPattern(v, m_ConstantFloat(&f)) && !f.isNaN() && !f.isNegative();
}

static bool isPositiveConst(Value v) {
  llvm::APFloat f(0.0f);
  return matchPattern(v, m_ConstantFloat(&f)) && !f.isNaN() && !f.isNegative() &&
         !f.isZero();
}

/// A float that cannot be negative, and cannot be a NaN either.
///
/// `arith.maxnumf(x, c)` with `c >= 0` is the relu `--select-to-minmax`
/// produces, and it hands back the operand that is **not** a NaN -- so its
/// result is at least `c` whatever `x` was. Rounding keeps that, and so does
/// multiplying by a positive constant, which is what the output scale is.
static bool nonNegative(Value v, unsigned depth = 0) {
  if (depth > 8)
    return false;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (llvm::isa<math::RoundEvenOp>(def))
    return nonNegative(def->getOperand(0), depth + 1);
  if (auto max = llvm::dyn_cast<arith::MaxNumFOp>(def))
    return isNonNegativeConst(max.getRhs()) || isNonNegativeConst(max.getLhs());
  if (auto mul = llvm::dyn_cast<arith::MulFOp>(def)) {
    if (isPositiveConst(mul.getRhs()))
      return nonNegative(mul.getLhs(), depth + 1);
    if (isPositiveConst(mul.getLhs()))
      return nonNegative(mul.getRhs(), depth + 1);
    return false;
  }
  if (auto div = llvm::dyn_cast<arith::DivFOp>(def))
    return isPositiveConst(div.getRhs()) &&
           nonNegative(div.getLhs(), depth + 1);
  return false;
}

/// **A quantizer below a relu does not need its lower clamp.**
///
/// A quantization tail is `clamp(round(x / s), -128, 127)`, and on a core with
/// no `Zbb` each end of that clamp is a compare, a branch and a `li`
/// ([[gemmlir-the-clamp-is-two-branches]]). Where the value came through a relu
/// it is already at least zero, so the lower end can never bite:
/// `arith.maxnumf(x, 0)` hands back the operand that is not a NaN, so its result
/// is non-negative whatever `x` was, and neither `math.roundeven` nor a multiply
/// by a positive scale can take it below zero.
///
/// This is not the rewrite that was measured and refused twice -- that one moved
/// the clamp into the float, which puts two more operations *on* the dependency
/// chain. This takes two instructions away and adds nothing.
///
/// **And a third refusal, 2026-09-15.** With the lower end gone, moving only the
/// *upper* one into the float is one `fmin.s` replacing one `blt` and one `li`,
/// on a body that by then had four chains to interleave -- the conditions that
/// had just reversed the relu relaxation
/// ([[gemmlir-the-chain-not-the-count]]). It lost anyway, and by more than
/// before: **+2.6% across the set**, every one of the thirteen slower,
/// `densenet121` +4.3% and `vit_tiny` +3.5%.
///
/// The difference from the relu, which won, is **what kind of branch is being
/// replaced**. A relu's is data-dependent and unpredictable -- about half the
/// values are negative -- so it costs a mispredict most times through. A clamp's
/// is almost never taken, because the scale is calibrated so the values fit, so
/// the predictor gets it right and it costs nothing. Trading an interleavable
/// mispredict for a chain step wins; trading a free branch for one loses.
///
/// An out-of-range value makes `arith.fptosi` poison, and `maxsi(poison, -128)`
/// is poison too, so dropping the maximum leaves the poison exactly where it
/// was.
class DropLowerClamp : public OpRewritePattern<arith::MaxSIOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MaxSIOp max,
                                PatternRewriter &rewriter) const final {
    Value value = max.getLhs(), bound = max.getRhs();
    llvm::APInt limit;
    if (!matchPattern(bound, m_ConstantInt(&limit))) {
      std::swap(value, bound);
      if (!matchPattern(bound, m_ConstantInt(&limit)))
        return failure();
    }
    // Only a bound at or below zero: a positive one is a real clamp.
    if (limit.isStrictlyPositive())
      return failure();

    auto convert = value.getDefiningOp<arith::FPToSIOp>();
    if (!convert || !nonNegative(convert.getIn()))
      return failure();

    rewriter.replaceOp(max, value);
    return success();
  }
};

class DropClampBelowRelu
    : public impl::DropClampBelowReluBase<DropClampBelowRelu> {
public:
  using impl::DropClampBelowReluBase<DropClampBelowRelu>::DropClampBelowReluBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<DropLowerClamp>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
