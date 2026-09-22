//===- UnrollAcceleratorLoopsPass.cpp -----------------------------*- C++ -*-===//
//
// A loop the flush analysis cannot see inside keeps every flush.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Matchers.h"

#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_UNROLLACCELERATORLOOPS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

static bool isAccelerator(Operation *op) {
  return llvm::isa<MatMulInt8Op, MatMulInt8ScaleOp, Conv2DInt8Op,
                   DepthwiseConv2DInt8Op, ResAddInt8Op, NormInt8Op>(op);
}

/// `--place-cache-flushes` models a region it cannot otherwise account for as
/// one host step that reads and writes everything inside, and **never gives the
/// accelerator calls in that region an attribute** -- so they keep both their
/// flushes, and a flush is a walk of the whole L1.
///
/// A transformer is the only shape in the set with such loops: a batch matmul
/// lowers to one slice a trip, and `vit_tiny` has **24 of them**, each three
/// iterations of nothing but a `gemmlir.matmul_i8`. `gemmlir_evict` is 6.0% of
/// that model by program-counter sampling.
///
/// Nothing in those loops is a host step at all. Unrolling them fully puts the
/// calls in the straight line the analysis already handles, where a flush
/// survives only where the host really did write in between.
///
/// Sound because the flush is not the fence: `gemmlir.no_flush_*` controls only
/// the cache walk, and the `fence` that waits for the accelerator to go idle is
/// emitted by the runtime on every call regardless
/// ([[gemmini-conv-does-not-fence]] is about the fence, and it stays).
class UnrollAcceleratorLoops
    : public impl::UnrollAcceleratorLoopsBase<UnrollAcceleratorLoops> {
public:
  using impl::UnrollAcceleratorLoopsBase<
      UnrollAcceleratorLoops>::UnrollAcceleratorLoopsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    SmallVector<scf::ForOp> candidates;
    getOperation().walk([&](scf::ForOp loop) {
      if (!loop.getResults().empty())
        return;
      std::optional<int64_t> lo = constantOf(loop.getLowerBound());
      std::optional<int64_t> hi = constantOf(loop.getUpperBound());
      std::optional<int64_t> st = constantOf(loop.getStep());
      if (!lo || !hi || !st || *st <= 0 || *hi <= *lo)
        return;
      int64_t trips = (*hi - *lo + *st - 1) / *st;
      if (trips < 2 || trips > maxTrips)
        return;
      // Only a loop whose every step is the accelerator's: one host operation
      // in there and unrolling buys nothing, because the analysis would keep
      // the flushes anyway.
      bool accelerator = false;
      bool clean = true;
      loop.getBody()->walk([&](Operation *op) {
        if (op == loop.getOperation() || llvm::isa<scf::YieldOp>(op))
          return WalkResult::advance();
        if (isAccelerator(op)) {
          accelerator = true;
          return WalkResult::advance();
        }
        if (isMemoryEffectFree(op) ||
            llvm::isa<memref::AllocOp, memref::AllocaOp, memref::DeallocOp>(op))
          return WalkResult::advance();
        clean = false;
        return WalkResult::interrupt();
      });
      if (accelerator && clean)
        candidates.push_back(loop);
    });
    for (scf::ForOp loop : candidates)
      if (failed(loopUnrollFull(loop)))
        signalPassFailure();
  }

private:
  static std::optional<int64_t> constantOf(Value v) {
    IntegerAttr attr;
    if (!matchPattern(v, m_Constant(&attr)))
      return std::nullopt;
    return attr.getInt();
  }
};

} // namespace

} // namespace mlir::gemmlir
