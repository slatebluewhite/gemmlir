//===- UnrollElementwiseLoopsPass.cpp ----------------------------*- C++ -*-===//
//
// Two elements a trip, so two dependency chains can overlap.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_UNROLLELEMENTWISELOOPS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The number of iterations, when all three bounds are constants.
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

/// The innermost loop of a conversion between f32 and i8 is a dozen
/// instructions of which the work is a load, a multiply, a convert and a store
/// -- a chain the core walks one instruction at a time, with nothing to put in
/// the gaps. Two iterations in one body gives the scheduler a second,
/// independent chain to interleave, and it is worth more than the loop
/// arithmetic it also removes:
///
/// | | cycles/element |
/// |---|---|
/// | the input quantize | 22.0 -> **19.0** |
/// | the dequantize tail | 26.2 -> **23.1** |
///
/// **Four, not two.** The measurement above said four was no better, and it was
/// right about the pipeline it was made on: the loops then still carried the
/// reduction accumulator through memory, ran their windows as loops, and spent
/// three operations an element on per-channel numbers. On the chains that are
/// left after all three of those went, four wins clearly -- `vit_tiny` -5.2%,
/// `googlenet` -4.6%, `densenet121` -4.4%, `efficientnet_b0` -3.8%, twelve of
/// thirteen faster and none slower than +0.1%, every model byte-identical.
///
/// **Eight where eight divides, four where it does not.** Eight is better than
/// four on every body the set runs -- measured on the board at the shapes the
/// models use, with the three forms checked byte for byte against each other
/// first:
///
/// | | x1 | x2 | x4 | x8 | x16 |
/// |---|---|---|---|---|---|
/// | DenseNet's integer batch norm | 22.4 | 19.5 | **17.5** | 16.8 | 16.5 |
/// | a float requantize of an i32 accumulator | 25.4 | 24.8 | **24.1** | 23.2 | 23.1 |
/// | EfficientNet's per-channel gate | 26.4 | 23.4 | **21.8** | 21.0 | 20.6 |
///
/// Eight is -3.7% to -3.8% against the four that shipped, and sixteen buys
/// another 0.3% to 1.9% for twice the code again -- not worth it against the
/// instruction fetch a 193 KB `forward` already pays.
///
/// The reason four shipped instead of eight was that the factor has to divide
/// the trip count, so eight reaches fewer loops. That is an argument for trying
/// eight **and then four**, not for picking one: a trip count of 32 takes
/// eight, a trip count of 12 takes four, a trip count of 6 takes two, and a
/// trip count of 17 takes none -- exactly as before.
///
/// Only a leaf: a loop with another loop inside it is the nest's scaffolding
/// and unrolling it duplicates the whole subtree, and a loop with a call in it
/// is an accelerator or a copy whose cost is not the loop. And only when the
/// factor divides the trip count, so there is no epilogue -- a second copy of
/// the body for the sake of a remainder that never happens here.
class UnrollElementwiseLoops
    : public impl::UnrollElementwiseLoopsBase<UnrollElementwiseLoops> {
public:
  using impl::UnrollElementwiseLoopsBase<
      UnrollElementwiseLoops>::UnrollElementwiseLoopsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    if (factor < 2)
      return;
    SmallVector<std::pair<scf::ForOp, unsigned>> leaves;
    getOperation().walk([&](scf::ForOp forOp) {
      bool skip = false;
      forOp.getBody()->walk([&](Operation *op) {
        if (llvm::isa<scf::ForOp, scf::WhileOp, scf::ParallelOp>(op) ||
            llvm::isa<CallOpInterface>(op))
          skip = true;
      });
      if (skip)
        return;
      std::optional<int64_t> trips = tripCount(forOp);
      if (!trips)
        return;
      // The largest factor that divides the trip count, so there is never an
      // epilogue. `factor` is the ceiling, not the only candidate.
      for (unsigned f = factor; f >= 2; f /= 2)
        if (*trips % f == 0 && *trips >= 2 * (int64_t)f) {
          leaves.emplace_back(forOp, f);
          break;
        }
    });
    for (auto [forOp, f] : leaves)
      (void)loopUnrollByFactor(forOp, f);
  }
};

} // namespace

} // namespace mlir::gemmlir
