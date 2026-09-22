//===- PlaceCacheFlushesPass.cpp -------------------------------*- C++ -*-===//
//
// Keep the cache flushes that are load-bearing and drop the rest.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"

#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_PLACECACHEFLUSHES
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

constexpr llvm::StringLiteral kNoFlushBefore = "gemmlir.no_flush_before";
constexpr llvm::StringLiteral kNoFlushAfter = "gemmlir.no_flush_after";

bool isAcceleratorOp(Operation *op) {
  return isa<MatMulInt8Op, MatMulInt8ScaleOp, ResAddInt8Op, Conv2DInt8Op,
             DepthwiseConv2DInt8Op>(op);
}

/// The buffer a view ultimately refers to. Two views of the same allocation
/// share a base even when their windows do not overlap -- that is the safe
/// direction, since a flush is all-or-nothing anyway.
Value baseOf(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto op = dyn_cast<memref::SubViewOp>(def)) { v = op.getSource(); continue; }
    if (auto op = dyn_cast<memref::ViewOp>(def)) { v = op.getSource(); continue; }
    if (auto op = dyn_cast<memref::CastOp>(def)) { v = op.getSource(); continue; }
    if (auto op = dyn_cast<memref::ReinterpretCastOp>(def)) { v = op.getSource(); continue; }
    if (auto op = dyn_cast<memref::ExpandShapeOp>(def)) { v = op.getSrc(); continue; }
    if (auto op = dyn_cast<memref::CollapseShapeOp>(def)) { v = op.getSrc(); continue; }
    break;
  }
  return v;
}

/// Buffers are identified by a small integer so the simulation below works on
/// bitsets. Every `memref.get_global` of one symbol is the same buffer however
/// many times it is fetched.
class Buffers {
public:
  /// -1 when the value is not a memref, which the caller ignores.
  int idOf(Value v) {
    if (!isa<MemRefType>(v.getType()))
      return -1;
    Value b = baseOf(v);
    if (auto g = b.getDefiningOp<memref::GetGlobalOp>()) {
      auto [it, inserted] = globals.try_emplace(g.getName(), (int)count);
      if (inserted)
        count++;
      return it->second;
    }
    auto [it, inserted] = values.try_emplace(b, (int)count);
    if (inserted)
      count++;
    return it->second;
  }
  size_t size() const { return count; }

private:
  llvm::DenseMap<Value, int> values;
  llvm::DenseMap<llvm::StringRef, int> globals;
  size_t count = 0;
};

/// One step of the simulation: which buffers this operation reads and writes,
/// and whether it runs on the host or on the accelerator.
struct Step {
  Operation *op = nullptr;
  bool accelerator = false;
  llvm::SmallVector<int, 4> reads, writes;
};

/// What the host's cache may hold. A flush is global, so these are plain sets.
struct CacheState {
  llvm::SmallVector<bool> dirty;   // the host wrote it and may still hold it
  llvm::SmallVector<bool> resident; // the host may hold lines of it
  llvm::SmallVector<bool> stale;   // ...and the accelerator has overwritten them

  explicit CacheState(size_t n) : dirty(n, false), resident(n, false), stale(n, false) {}
  void flush() {
    dirty.assign(dirty.size(), false);
    resident.assign(resident.size(), false);
    stale.assign(stale.size(), false);
  }
  bool operator==(const CacheState &o) const {
    return dirty == o.dirty && resident == o.resident && stale == o.stale;
  }
};

class PlaceCacheFlushes
    : public impl::PlaceCacheFlushesBase<PlaceCacheFlushes> {
public:
  using impl::PlaceCacheFlushesBase<PlaceCacheFlushes>::PlaceCacheFlushesBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (func.isExternal() || !func.getBody().hasOneBlock())
      return;
    Block &block = func.getBody().front();

    Buffers buffers;
    llvm::SmallVector<Step> steps;
    llvm::SmallVector<int> escaping;

    // Whatever the caller handed in, the caller also wrote and may still hold.
    for (BlockArgument arg : block.getArguments()) {
      int id = buffers.idOf(arg);
      if (id >= 0)
        escaping.push_back(id);
    }

    for (Operation &op : block) {
      Step step;
      step.op = &op;
      if (isAcceleratorOp(&op)) {
        step.accelerator = true;
        if (!describeAccelerator(&op, buffers, step))
          return; // an operand shape this pass does not model: keep every flush
        steps.push_back(step);
        continue;
      }
      if (isa<func::ReturnOp>(op)) {
        // The caller reads what comes back.
        for (Value v : op.getOperands()) {
          int id = buffers.idOf(v);
          if (id >= 0)
            escaping.push_back(id);
        }
        continue;
      }
      if (isMemoryEffectFree(&op) || isa<memref::AllocOp, memref::AllocaOp,
                                        memref::DeallocOp>(op))
        continue;
      auto effects = dyn_cast<MemoryEffectOpInterface>(&op);
      if (!effects) {
        // An operation that carries a region and declares no effects of its own
        // -- `scf.for` is the one that matters -- used to end the analysis here,
        // and with it every flush in the function. A transformer is 24 such
        // loops (a batch matmul, one slice a trip), so `vit_tiny` kept **all**
        // 124 of its flushes and this pass did nothing at all for it.
        //
        // It is enough to treat the loop as one host step that reads and writes
        // every buffer anything inside it touches. Conservative on both counts:
        // a read makes a stale buffer fail the check, and a write makes it
        // dirty for whatever comes after. The accelerator calls **inside** the
        // region are never given an attribute, so they keep both their flushes
        // -- which is what makes it sound to say nothing inside is left stale.
        //
        // `vit_tiny` 360.68 -> **355.28** ms, -1.5%: fifty `after` flushes of
        // its 124 go, and a flush is a walk of the whole L1. Byte for byte over
        // forty runs against both references -- which is the check that matters
        // for this pass, because the failure it can cause shows up in about one
        // run in four and depends on what ran before. `convnext_tiny`, which is
        // 36 such loops, is the other model this reaches.
        if (op.getNumRegions() == 0)
          return; // cannot see what it touches: keep every flush
        llvm::SmallVector<int, 8> touched;
        bool modelled = true;
        op.walk([&](Operation *inner) {
          if (inner == &op)
            return WalkResult::advance();
          Step nested;
          if (isAcceleratorOp(inner)) {
            if (!describeAccelerator(inner, buffers, nested))
              modelled = false;
          } else if (isMemoryEffectFree(inner) ||
                     isa<memref::AllocOp, memref::AllocaOp, memref::DeallocOp>(
                         inner)) {
            return WalkResult::advance();
          } else if (auto e = dyn_cast<MemoryEffectOpInterface>(inner)) {
            llvm::SmallVector<MemoryEffects::EffectInstance> inners;
            e.getEffects(inners);
            for (auto &x : inners) {
              if (!x.getValue()) {
                modelled = false;
                break;
              }
              int id = buffers.idOf(x.getValue());
              if (id >= 0)
                nested.reads.push_back(id);
            }
          } else if (inner->getNumRegions() == 0) {
            modelled = false;
          }
          if (!modelled)
            return WalkResult::interrupt();
          touched.append(nested.reads.begin(), nested.reads.end());
          touched.append(nested.writes.begin(), nested.writes.end());
          return WalkResult::advance();
        });
        if (!modelled)
          return;
        step.reads.assign(touched.begin(), touched.end());
        step.writes.assign(touched.begin(), touched.end());
        if (!step.reads.empty())
          steps.push_back(step);
        continue;
      }
      llvm::SmallVector<MemoryEffects::EffectInstance> list;
      effects.getEffects(list);
      for (auto &e : list) {
        Value v = e.getValue();
        if (!v) // an effect on no particular value: assume the worst
          return;
        int id = buffers.idOf(v);
        if (id < 0)
          continue;
        if (isa<MemoryEffects::Read>(e.getEffect()))
          step.reads.push_back(id);
        else if (isa<MemoryEffects::Write>(e.getEffect()))
          step.writes.push_back(id);
      }
      if (!step.reads.empty() || !step.writes.empty())
        steps.push_back(step);
    }

    // The buffers the caller can see are read by the host after the last step
    // and written by it before the first, every time the function runs.
    Step caller;
    caller.reads.assign(escaping.begin(), escaping.end());
    caller.writes.assign(escaping.begin(), escaping.end());

    // Start from what the runtime does today and take away only what can be
    // taken away: a flush stays unless the function still checks out without it.
    llvm::SmallVector<bool> before(steps.size(), true), after(steps.size(), true);
    for (size_t i = 0; i < steps.size(); i++) {
      if (!steps[i].accelerator)
        continue;
      if (dropBefore) {
        before[i] = false;
        if (!holds(steps, caller, buffers.size(), before, after))
          before[i] = true;
      }
      if (dropAfter) {
        after[i] = false;
        if (!holds(steps, caller, buffers.size(), before, after))
          after[i] = true;
      }
    }

    for (size_t i = 0; i < steps.size(); i++) {
      if (!steps[i].accelerator)
        continue;
      if (!before[i])
        steps[i].op->setAttr(kNoFlushBefore, UnitAttr::get(&getContext()));
      if (!after[i])
        steps[i].op->setAttr(kNoFlushAfter, UnitAttr::get(&getContext()));
    }
  }

private:
  /// The accelerator ops all read their inputs and write one output; a matmul
  /// that accumulates reads its output as well, because that is the D operand.
  static bool describeAccelerator(Operation *op, Buffers &buffers, Step &step) {
    auto add = [&](Value v, bool write) {
      int id = buffers.idOf(v);
      if (id < 0)
        return;
      (write ? step.writes : step.reads).push_back(id);
    };
    if (auto mm = dyn_cast<MatMulInt8Op>(op)) {
      add(mm.getLhsMat(), false);
      add(mm.getRhsMat(), false);
      if (mm.getBias())
        add(mm.getBias(), false);
      if (mm.getAccumulate())
        add(mm.getOutMat(), false);
      add(mm.getOutMat(), true);
      return true;
    }
    if (auto mm = dyn_cast<MatMulInt8ScaleOp>(op)) {
      add(mm.getLhsMat(), false);
      add(mm.getRhsMat(), false);
      if (mm.getBias())
        add(mm.getBias(), false);
      add(mm.getOutMat(), true);
      return true;
    }
    if (auto ra = dyn_cast<ResAddInt8Op>(op)) {
      add(ra.getLhsMat(), false);
      add(ra.getRhsMat(), false);
      add(ra.getOutMat(), true);
      return true;
    }
    if (auto cv = dyn_cast<Conv2DInt8Op>(op)) {
      add(cv.getInput(), false);
      add(cv.getFilter(), false);
      if (cv.getBias())
        add(cv.getBias(), false);
      add(cv.getOutput(), true);
      return true;
    }
    if (auto cv = dyn_cast<DepthwiseConv2DInt8Op>(op)) {
      add(cv.getInput(), false);
      add(cv.getFilter(), false);
      if (cv.getBias())
        add(cv.getBias(), false);
      add(cv.getOutput(), true);
      return true;
    }
    return false;
  }

  /// Run the function until the cache state repeats, and report whether any
  /// accelerator call read a buffer the host still held dirty, or any host read
  /// returned lines the accelerator had overwritten.
  ///
  /// Running it more than once is the point. The function is called in a loop
  /// and the host's lines outlive one call, so a flush that looks unnecessary
  /// within a single pass through the body can still be the one keeping the
  /// *next* call honest.
  static bool holds(llvm::ArrayRef<Step> steps, const Step &caller, size_t n,
                    llvm::ArrayRef<bool> before, llvm::ArrayRef<bool> after) {
    CacheState state(n);
    llvm::SmallVector<CacheState, 4> seen;
    for (int round = 0; round < 16; round++) {
      for (CacheState &s : seen)
        if (s == state)
          return true; // the cycle is closed and nothing went wrong in it
      seen.push_back(state);

      // Entering the function: the caller wrote the arguments.
      for (int id : caller.writes) {
        state.dirty[id] = true;
        state.resident[id] = true;
        state.stale[id] = false;
      }
      for (size_t i = 0; i < steps.size(); i++) {
        const Step &step = steps[i];
        if (step.accelerator) {
          if (before[i])
            state.flush();
          for (int id : step.reads) {
            if (state.dirty[id])
              return false; // reads past the host's stores
            // The runtime touches every operand on the **host** before the call
            // -- `gemmlir_first_read` reads a byte a page so the accelerator
            // does not meet a page with no PTE and read it as zeros. That is a
            // host read like any other: it pulls lines in, and if the
            // accelerator has since overwritten the buffer, the lines it pulls
            // are the stale ones.
            //
            // Leaving it out of the model is what let this pass take away a
            // flush it needed. The symptom was an answer that depended on what
            // program had run *before*, about one run in four -- a big model
            // first is what makes the host hold old lines for the address at
            // all -- and it went away under any instrumentation, because
            // reading the buffer to check it is itself the missing read.
            if (state.stale[id])
              return false;
            state.resident[id] = true;
          }
          for (int id : step.writes) {
            state.dirty[id] = false;
            if (state.resident[id])
              state.stale[id] = true;
          }
          if (after[i])
            state.flush();
          continue;
        }
        for (int id : step.reads) {
          if (state.stale[id])
            return false; // the host reads what the accelerator replaced
          state.resident[id] = true;
        }
        for (int id : step.writes) {
          state.dirty[id] = true;
          state.resident[id] = true;
          state.stale[id] = false;
        }
      }
      // Leaving it: the caller reads whatever it can see.
      for (int id : caller.reads) {
        if (state.stale[id])
          return false;
        state.resident[id] = true;
      }
    }
    return false; // no cycle found in time: do not take the flush away
  }
};

} // namespace

} // namespace mlir::gemmlir
