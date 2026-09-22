//===- ApproximateExpPass.cpp ------------------------------------*- C++ -*-===//
//
// A softmax's exponential does not need libm.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_APPROXIMATEEXP
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// `math.exp` lowers to a call to libm's `expf`, which is correctly rounded and
/// carries `errno`: the profile of a decoder-only transformer shows not only
/// `expf32` but `xflowf`, `with_errnof` and `__math_uflowf` -- the error paths.
/// Together **9.9% of the model**, for 3.3% of its elements, which works out at
/// about 57 cycles an exponential.
///
/// A softmax's exponential is divided by the sum of its siblings and the
/// quotient is quantized to an i8, so the last bits of `expf` are thrown away
/// twice over. MLIR's own polynomial approximation is a range reduction and a
/// degree-6 polynomial -- accurate to a few ULP, not a crude approximation --
/// and it has no call, no `errno` and no error paths.
///
/// Only `math.exp`: `--convert-math-to-libm` still takes everything else, and
/// the one transcendental this pipeline has left in a hot loop is this one.
///
/// **And it loses. Not in the pipeline.**
///
/// | | ms | |
/// |---|---|---|
/// | `gpt_tiny` | 566.46 -> 594.72 | **+5.0%** |
/// | `regnet_y_400mf` | 145.24 -> 147.73 | +1.7% |
/// | `vit_tiny` | 355.29 -> 357.36 | +0.6% |
///
/// Every model it reaches got slower, the one with the most exponentials most
/// of all. Relative L2 did not move on any of them, and the final outputs were
/// byte for byte identical -- the difference is below the i8 the softmax is
/// quantized to, which is what the licence said it would be. The licence was
/// right; the arithmetic was not worth it.
///
/// Why: this board's `expf` is a small table and a short polynomial, and the
/// call pipelines. The approximation is 24 operations of which nine are an
/// `fma` on the previous one, plus a `math.floor` that expands inline and four
/// `select`s -- one long dependency chain on an in-order single-issue core,
/// which is the thing that costs here ([[gemmlir-the-chain-not-the-count]]).
///
/// Kept because it is one line to switch on and the answer depends on the
/// libm: a build whose `expf` goes through a slow generic path would want it.
class ApproximateExp : public impl::ApproximateExpBase<ApproximateExp> {
public:
  using impl::ApproximateExpBase<ApproximateExp>::ApproximateExpBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, math::MathDialect,
                    vector::VectorDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    populateMathPolynomialApproximationPatterns(
        patterns, [](StringRef name) { return name == math::ExpOp::getOperationName(); });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
