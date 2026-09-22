//===- PromoteReductionAccumulatorPass.cpp ----------------------*- C++ -*-===//
//
// Carry a reduction's running value in a register instead of a memref slot.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_PROMOTEREDUCTIONACCUMULATOR
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// True when every index, and the buffer itself, is fixed for the whole loop.
static bool fixedForTheLoop(scf::ForOp forOp, memref::LoadOp load) {
  if (!forOp.isDefinedOutsideOfLoop(load.getMemRef()))
    return false;
  for (Value i : load.getIndices())
    if (!forOp.isDefinedOutsideOfLoop(i))
      return false;
  return true;
}

/// The one `memref.load` and the one `memref.store` that name the same slot,
/// or nothing. Both have to sit directly in the loop's own body: a store under
/// an `scf.if` updates the slot only sometimes, and hoisting it out of the
/// memory would drop the other branch's answer.
static std::pair<memref::LoadOp, memref::StoreOp>
theAccumulator(scf::ForOp forOp) {
  memref::LoadOp load;
  memref::StoreOp store;
  Block *body = forOp.getBody();

  for (Operation &op : *body) {
    if (auto s = llvm::dyn_cast<memref::StoreOp>(&op)) {
      if (store)
        return {}; // two stores: which one is the accumulator is not decidable
      store = s;
    }
  }
  if (!store)
    return {};

  // Nothing else in the loop -- at any depth -- may write memory. A call is
  // opaque and could write anything; another store could alias the slot. Reads
  // are fine: they cannot invalidate a value we are about to keep in a
  // register.
  WalkResult walk = body->walk([&](Operation *op) {
    if (op == store.getOperation() || llvm::isa<memref::LoadOp>(op))
      return WalkResult::advance();
    if (llvm::isa<CallOpInterface>(op))
      return WalkResult::interrupt();
    if (op->getNumRegions() > 0)
      return WalkResult::advance(); // a container; its body is walked too
    if (!isMemoryEffectFree(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return {};

  // The load that reads the slot the store writes, and the slot must be named
  // *only* by those two: another read of the same buffer inside the loop would
  // see the value we stopped writing.
  for (Operation &op : *body) {
    auto l = llvm::dyn_cast<memref::LoadOp>(&op);
    if (!l || l.getMemRef() != store.getMemRef() ||
        l.getIndices() != store.getIndices())
      continue;
    if (load)
      return {};
    load = l;
  }
  if (!load || !load->isBeforeInBlock(store) || !fixedForTheLoop(forOp, load))
    return {};

  unsigned uses = 0;
  for (Operation *user : store.getMemRef().getUsers())
    if (forOp->isAncestor(user))
      uses++;
  if (uses != 2)
    return {};
  return {load, store};
}

/// `--convert-linalg-to-loops` writes a reduction's running value back to its
/// output buffer every step, and reads it again on the next one:
///
/// ```
///   lb   a4, 0(s1)       # the element
///   lb   a1, 832(a2)     # the running maximum, from memory
///   blt  a4, a1, .store
///   mv   a1, a4
/// .store:
///   sb   a1, 832(a2)     # and back to memory
/// ```
///
/// On this in-order core that store-to-load round trip is the whole loop: a
/// 3x3 max-pool pays nine of them per output where one store would do. Nothing
/// downstream removes it, because `llc` runs no IR pipeline at all
/// ([[gemmlir-llc-is-not-minus-o2]]) -- there is no pass to promote the slot.
///
/// Carrying it as an `scf.for` iteration argument is the same computation with
/// the memory taken out of the middle. It also cascades: promoting the innermost
/// loop leaves the load and the store in the loop above, which is then itself a
/// candidate, so a 3x3 window ends up with one load before the nest and one
/// store after it.
class PromoteReductionAccumulator
    : public impl::PromoteReductionAccumulatorBase<
          PromoteReductionAccumulator> {
public:
  using impl::PromoteReductionAccumulatorBase<
      PromoteReductionAccumulator>::PromoteReductionAccumulatorBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect, memref::MemRefDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    bool again = true;
    // Innermost first, and then again: each promotion exposes the loop above.
    while (again) {
      again = false;
      SmallVector<scf::ForOp> loops;
      // `walk` is post-order: innermost first, which is the order the cascade
      // wants -- promoting the inner loop is what leaves the load and the store
      // in the loop above for the next one to find.
      getOperation().walk([&](scf::ForOp f) { loops.push_back(f); });
      for (scf::ForOp forOp : loops)
        if (promote(forOp, rewriter))
          again = true;
    }
  }

private:
  static bool promote(scf::ForOp forOp, IRRewriter &rewriter) {
    auto [load, store] = theAccumulator(forOp);
    if (!load || !store)
      return false;

    Location loc = forOp.getLoc();
    rewriter.setInsertionPoint(forOp);
    Value init = rewriter.create<memref::LoadOp>(loc, load.getMemRef(),
                                                 load.getIndices());

    Value stored = store.getValueToStore();
    FailureOr<LoopLikeOpInterface> replaced = forOp.replaceWithAdditionalYields(
        rewriter, ValueRange{init}, /*replaceInitOperandUsesInLoop=*/false,
        [&](OpBuilder &, Location, ArrayRef<BlockArgument>) {
          return SmallVector<Value>{stored};
        });
    if (failed(replaced)) {
      rewriter.eraseOp(init.getDefiningOp());
      return false;
    }

    auto newFor = llvm::cast<scf::ForOp>(replaced->getOperation());
    rewriter.replaceAllUsesWith(load.getResult(),
                                newFor.getRegionIterArgs().back());
    rewriter.eraseOp(load);
    rewriter.eraseOp(store);
    rewriter.setInsertionPointAfter(newFor);
    rewriter.create<memref::StoreOp>(loc, newFor.getResults().back(),
                                     init.getDefiningOp()->getOperand(0),
                                     init.getDefiningOp()->getOperands().drop_front());
    return true;
  }
};

} // namespace

} // namespace mlir::gemmlir
