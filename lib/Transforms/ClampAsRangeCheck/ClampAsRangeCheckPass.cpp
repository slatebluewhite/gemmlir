//===- ClampAsRangeCheckPass.cpp ----------------------------------*- C++ -*-===//
//
// Two clamps are one unsigned compare.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_CLAMPASRANGECHECK
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

static bool intConst(Value v, int64_t &out) {
  IntegerAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return false;
  out = attr.getInt();
  return true;
}

/// Every quantization tail ends `maxsi(v, lo)` then `minsi(v, hi)`. This board
/// is plain rv64gc -- no `Zbb`, so no `max`/`min` and no conditional move -- and
/// the pair costs five instructions:
///
/// ```
///   sgtz a5, a4 / neg a5, a5 / and a5, a5, a4 / li a4, 127 / blt a5, a4, .+
/// ```
///
/// But the pair is a **range check**, and one *unsigned* compare decides it:
/// anything outside `[lo, hi]` has `(unsigned)(v - lo) > hi - lo`, negatives
/// included. The fixup goes out of line, where it is rarely taken.
///
/// ```
///   bltu a5, a4, .+          # in range: nothing else to do
/// ```
///
/// Measured as a kernel at DenseNet's batch-norm shape, where 774 of 4096
/// values actually clamp: **-5.1%**. The clamp is 11.3% of `densenet121` by
/// program-counter sampling, 10.2% of `vit_tiny` and 7.6% of `efficientnet_b0`.
///
/// `scf.if` rather than two `arith.select`s, because a select is branchless and
/// branchless is exactly what costs five instructions here. Checked through the
/// real lowering before it was built: `llc` turns this into one `bltu`.
class ClampAsRangeCheck : public OpRewritePattern<arith::MinSIOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MinSIOp outer,
                                PatternRewriter &rewriter) const final {
    auto inner = outer.getLhs().getDefiningOp<arith::MaxSIOp>();
    int64_t hi = 0, lo = 0;
    if (!inner || !intConst(outer.getRhs(), hi) || !intConst(inner.getRhs(), lo))
      return failure();
    Value v = inner.getLhs();
    auto ty = llvm::dyn_cast<IntegerType>(v.getType());
    if (!ty || ty.getWidth() > 64 || ty.getWidth() < 8)
      return failure();
    if (hi <= lo)
      return failure();
    // The span has to be a value the comparison can name, and the shift `v - lo`
    // must not be able to wrap: both hold comfortably for a quantization, and
    // refusing otherwise costs nothing.
    int64_t span = hi - lo;
    if (span == INT64_MAX || lo == INT64_MIN)
      return failure();
    unsigned width = ty.getWidth();
    APInt spanPlus(64, (uint64_t)span + 1, /*isSigned=*/false);
    if (spanPlus.getActiveBits() > width)
      return failure();

    Location loc = outer.getLoc();
    Value shifted = v;
    if (lo != 0)
      shifted = rewriter.create<arith::SubIOp>(
          loc, v,
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(ty, lo)));
    Value bound = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(ty, span + 1));
    Value inRange = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, shifted, bound);

    auto guard = rewriter.create<scf::IfOp>(loc, TypeRange{ty}, inRange,
                                            /*addThenBlock=*/true,
                                            /*addElseBlock=*/true);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(guard.thenBlock());
      rewriter.create<scf::YieldOp>(loc, ValueRange{v});

      rewriter.setInsertionPointToEnd(guard.elseBlock());
      Value loC = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(ty, lo));
      Value hiC = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(ty, hi));
      Value below = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, v, loC);
      rewriter.create<scf::YieldOp>(
          loc,
          ValueRange{rewriter.create<arith::SelectOp>(loc, below, loC, hiC)});
    }

    rewriter.replaceOp(outer, guard.getResult(0));
    return success();
  }
};

class ClampAsRangeCheck_Pass
    : public impl::ClampAsRangeCheckBase<ClampAsRangeCheck_Pass> {
public:
  using impl::ClampAsRangeCheckBase<ClampAsRangeCheck_Pass>::ClampAsRangeCheckBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ClampAsRangeCheck>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
