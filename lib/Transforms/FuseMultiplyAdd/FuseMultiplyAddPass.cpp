//===- FuseMultiplyAddPass.cpp -----------------------------------*- C++ -*-===//
//
// A multiply feeding an add is one instruction on this core, not two.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FUSEMULTIPLYADD
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// `a * b + c` becomes `math.fma a, b, c`, which is `fmadd.s`.
///
/// The dequantize tail every unfoldable convolution leaves behind is
/// `sitofp`, `mulf` by the scale, `addf` the bias, then the activation. The
/// multiply and the add are each a separate trip through the FPU and the add
/// waits for the multiply, so on this in-order core they cost two latencies.
/// `fmadd.s` costs one. Measured on a 2048-element tail loop: **30.4 -> 26.3
/// cycles an element**.
///
/// Run **after** `--convert-linalg-to-gemmlir`: it reads exactly this
/// `mulf`/`addf` shape to recognise a dequantization it can fold into the
/// accelerator's own scaling, and would stop seeing it.
///
/// **It is not bit-exact, and it is more accurate.** The fused form rounds
/// once, on the sum, instead of twice; the unfused one throws away the low
/// half of the product before adding. That is a change to what the tail
/// computes, of the same kind and size as multiplying by a scale's reciprocal
/// rather than dividing -- and, like it, one that leaves the model's distance
/// from the reference where it was.
class MultiplyThenAdd : public OpRewritePattern<arith::AddFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp add,
                                PatternRewriter &rewriter) const final {
    // Only one of the two can be folded in, and a multiply with another reader
    // would have to stay behind anyway -- fusing it then costs an instruction
    // instead of saving one.
    auto usable = [](Value v) {
      auto mul = v.getDefiningOp<arith::MulFOp>();
      return mul && mul->hasOneUse() ? mul : arith::MulFOp();
    };
    arith::MulFOp mul = usable(add.getLhs());
    Value addend = add.getRhs();
    if (!mul) {
      mul = usable(add.getRhs());
      addend = add.getLhs();
    }
    if (!mul)
      return failure();

    rewriter.replaceOpWithNewOp<math::FmaOp>(add, mul.getLhs(), mul.getRhs(),
                                             addend);
    return success();
  }
};

class FuseMultiplyAdd : public impl::FuseMultiplyAddBase<FuseMultiplyAdd> {
public:
  using impl::FuseMultiplyAddBase<FuseMultiplyAdd>::FuseMultiplyAddBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, math::MathDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MultiplyThenAdd>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
