//===- UnrollReductionWindowsPass.cpp ---------------------------*- C++ -*-===//
//
// A three-iteration window is more loop than work; straighten it out.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_UNROLLREDUCTIONWINDOWS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

static std::optional<int64_t> tripCount(scf::ForOp forOp) {
  llvm::APInt lb, ub, step;
  if (!matchPattern(forOp.getLowerBound(), m_ConstantInt(&lb)) ||
      !matchPattern(forOp.getUpperBound(), m_ConstantInt(&ub)) ||
      !matchPattern(forOp.getStep(), m_ConstantInt(&step)))
    return std::nullopt;
  int64_t l = lb.getSExtValue(), u = ub.getSExtValue(), s = step.getSExtValue();
  if (s <= 0 || u <= l)
    return std::nullopt;
  return (u - l + s - 1) / s;
}

static int64_t bodySize(scf::ForOp forOp) {
  int64_t n = 0;
  forOp.getBody()->walk([&](Operation *) { n++; });
  return n;
}

/// After [[gemmlir-the-accumulator-lives-in-memory]] takes the store-to-load
/// round trip out of a reduction, what is left in GoogLeNet's hot loops is
/// almost pure `addi`/`li`/`add`/`blt`/`lui`: the **nest scaffolding**. A 3x3
/// max-pool is a six-deep nest whose innermost loop runs three times, so the
/// bookkeeping costs more than the nine loads and eight compares it is there to
/// schedule.
///
/// Unrolling those two levels away leaves one straight run of loads and
/// compares, and because the accumulator is already an iteration argument the
/// run is a single chain with no memory in it.
///
/// Only a loop that **carries a value** -- one that `scf.for` gives a result --
/// so this stays on reductions, where the trip counts are window sizes. A plain
/// elementwise leaf runs the length of a row and belongs to
/// `--unroll-elementwise-loops`, which found two iterations to be the best
/// factor there.
///
/// The budget is on the *unrolled* body, and it is what stops a 7x7 window from
/// unrolling its outer level: 3x3 straightens completely, 7x7 straightens the
/// inner one and keeps the loop above it.
class UnrollReductionWindows
    : public impl::UnrollReductionWindowsBase<UnrollReductionWindows> {
public:
  using impl::UnrollReductionWindowsBase<
      UnrollReductionWindows>::UnrollReductionWindowsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    bool again = true;
    while (again) {
      again = false;
      SmallVector<scf::ForOp> loops;
      // `walk` is post-order, so this is already innermost first -- and the
      // order is the budget: straightening `kw` is what makes `kh`'s body too
      // big to straighten, which is how a 7x7 window keeps its outer loop.
      // Reversing it (the natural-looking mistake) unrolls outside in, where
      // every level still looks small, and a 7x7 goes all the way to 49 loads.
      getOperation().walk([&](scf::ForOp f) { loops.push_back(f); });
      for (scf::ForOp forOp : loops) {
        if (forOp.getNumResults() == 0)
          continue;
        std::optional<int64_t> trips = tripCount(forOp);
        if (!trips || *trips < 2 || *trips > (int64_t)maxTrips)
          continue;
        if (bodySize(forOp) * *trips > (int64_t)maxOps)
          continue;
        bool opaque = false;
        forOp.getBody()->walk([&](Operation *op) {
          if (llvm::isa<CallOpInterface>(op))
            opaque = true;
        });
        if (opaque)
          continue;
        if (succeeded(loopUnrollFull(forOp)))
          again = true;
      }
    }
  }
};

} // namespace

} // namespace mlir::gemmlir
