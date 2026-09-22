//===- SaturateConstantCastsPass.cpp -----------------------------*- C++ -*-===//
//
// A constant that does not fit the integer it is converted to.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_SATURATECONSTANTCASTS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// **A max-pool's padding is minus infinity, and quantizing it is undefined.**
///
/// A padded max-pool arrives from the frontend with `-inf` in the border. When
/// the pool is moved onto i8 the quantization goes with it, and what is left in
/// front of the pool is
///
/// ```
///   %135 = arith.fptosi %cst : f32 to i32        // %cst is 0xFF800000
///   %136 = arith.maxsi  %135, -128
///   %137 = arith.minsi  %136, 127
///   %138 = arith.trunci %137 : i32 to i8         // and this fills the border
/// ```
///
/// `arith.fptosi` of a value outside the destination's range is **poison**, and
/// `maxsi(poison, -128)` is poison too, so the clamp does not rescue it. MLIR's
/// folder leaves the operation alone for exactly that reason, and what the
/// border ends up holding is then whatever the backend happens to materialize
/// -- which **changes with the surrounding code**. GoogLeNet's answer moved by
/// 0.6% of its output range when `--unroll-elementwise-loops` was raised from
/// two to four, and this one byte is why.
///
/// Replacing a poison value with a defined one is a refinement, so folding the
/// conversion is always legal; folding it the way the hardware does -- saturate
/// to the destination's extremes, and hand back the maximum for a NaN, which is
/// what RISC-V's `fcvt.w.s` gives -- is the choice that makes the compiled
/// answer and the obvious reading of the source agree. For a max-pool's border
/// that is `INT_MIN`, which the clamp below turns into the i8 minimum: the
/// pool's own identity, which is what the padding was always meant to be.
class SaturateConstantCast : public OpRewritePattern<arith::FPToSIOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::FPToSIOp cast,
                                PatternRewriter &rewriter) const final {
    auto intTy = llvm::dyn_cast<IntegerType>(cast.getType());
    if (!intTy || intTy.getWidth() > 64)
      return failure();
    llvm::APFloat value(0.0f);
    if (!matchPattern(cast.getIn(), m_ConstantFloat(&value)))
      return failure();

    unsigned width = intTy.getWidth();
    llvm::APInt folded(width, 0);
    if (value.isNaN()) {
      folded = llvm::APInt::getSignedMaxValue(width);
    } else {
      bool exact = false;
      llvm::APSInt out(width, /*isUnsigned=*/false);
      llvm::APFloat::opStatus status =
          value.convertToInteger(out, llvm::APFloat::rmTowardZero, &exact);
      if (status == llvm::APFloat::opInvalidOp)
        folded = value.isNegative() ? llvm::APInt::getSignedMinValue(width)
                                    : llvm::APInt::getSignedMaxValue(width);
      else
        folded = out;
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        cast, intTy, rewriter.getIntegerAttr(intTy, folded));
    return success();
  }
};

class SaturateConstantCasts
    : public impl::SaturateConstantCastsBase<SaturateConstantCasts> {
public:
  using impl::SaturateConstantCastsBase<
      SaturateConstantCasts>::SaturateConstantCastsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, arith::ArithDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SaturateConstantCast>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
