//===- ReciprocalForDivisionPass.cpp -----------------------------*- C++ -*-===//
//
// Dividing by a constant is multiplying by its reciprocal.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_RECIPROCALFORDIVISION
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// `x / c` becomes `x * (1/c)` for a constant `c`.
///
/// Measured on the board: the divide is **22 cycles an element** of a
/// quantization loop that is otherwise about thirteen instructions -- on this
/// in-order core `fdiv.s` is most of the loop. Every quantization in the
/// pipeline divides by its scale.
///
/// This must run **after** `--convert-linalg-to-gemmlir`: the requantization
/// matchers read the `divf` to recover the scale, and they would stop seeing it.
///
/// Not bit-exact against the division: `1/c` is rounded once, so the product
/// can differ from the quotient by an ulp and, where that lands on a tie, the
/// rounding after it can differ by one. That is a change to the quantization's
/// own definition rather than an error in it -- the scale is a measured
/// statistic and a reciprocal of it is as good a multiplier -- and both
/// runtimes compile from the same object, so the accelerator and the host still
/// agree byte for byte.
class DivideByConstant : public OpRewritePattern<arith::DivFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp div,
                                PatternRewriter &rewriter) const final {
    auto type = llvm::dyn_cast<FloatType>(div.getType());
    if (!type)
      return failure();
    llvm::APFloat c(0.0f);
    if (!matchPattern(div.getRhs(), m_ConstantFloat(&c)))
      return failure();
    if (!c.isFiniteNonZero() || c.isNegative())
      return failure();

    // The reciprocal being *inexact* is the point -- that ulp is what is being
    // traded for the divide. What it must not be is a different kind of number:
    // one over a huge scale is denormal or zero, and multiplying by that would
    // flush ordinary values away rather than round them.
    llvm::APFloat inv(c.getSemantics(), 1);
    inv.divide(c, llvm::APFloat::rmNearestTiesToEven);
    if (!inv.isFiniteNonZero() || inv.isDenormal())
      return failure();

    Value reciprocal = rewriter.create<arith::ConstantOp>(
        div.getLoc(), type, rewriter.getFloatAttr(type, inv));
    rewriter.replaceOpWithNewOp<arith::MulFOp>(div, div.getLhs(), reciprocal);
    return success();
  }
};

class ReciprocalForDivision
    : public impl::ReciprocalForDivisionBase<ReciprocalForDivision> {
public:
  using impl::ReciprocalForDivisionBase<
      ReciprocalForDivision>::ReciprocalForDivisionBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<DivideByConstant>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
