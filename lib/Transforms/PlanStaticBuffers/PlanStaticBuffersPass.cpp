//===- PlanStaticBuffersPass.cpp -------------------*- C++ -*-===//
//
// Gives the function's temporaries fixed addresses, laid out in one arena.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

#include "Gemmlir/GemmlirDialect.h"
#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_PLANSTATICBUFFERS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// Views onto a buffer keep it alive; a value that leaves the function through
/// one of them is the caller's.
bool isView(Operation *op) {
  return llvm::isa<memref::SubViewOp, memref::CollapseShapeOp,
                   memref::ExpandShapeOp, memref::ReinterpretCastOp,
                   memref::CastOp>(op);
}

/// One buffer to place: how big it is, and between which two operation indices
/// anything can still read it.
struct Slot {
  memref::AllocOp alloc;
  int64_t size = 0;
  int64_t align = 64;
  int64_t offset = 0;
};

int64_t roundUp(int64_t v, int64_t to) { return (v + to - 1) / to * to; }

class PlanStaticBuffers
    : public impl::PlanStaticBuffersBase<PlanStaticBuffers> {
public:
  using impl::PlanStaticBuffersBase<
      PlanStaticBuffers>::PlanStaticBuffersBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getBodyRegion());
    unsigned counter = 0;

    module.walk([&](func::FuncOp func) {
      if (func.isExternal())
        return;
      SmallVector<Slot> slots;
      func.walk([&](memref::AllocOp alloc) {
        auto type = llvm::dyn_cast<MemRefType>(alloc.getType());
        if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity())
          return;
        if (!alloc.getDynamicSizes().empty())
          return;
        // Only the allocations the function itself makes: one inside a region
        // lives once per iteration of whatever owns that region, which the
        // positions below do not describe.
        if (alloc->getParentOp() != func.getOperation())
          return;
        unsigned width = type.getElementTypeBitWidth();
        if (width == 0 || width % 8 != 0)
          return;

        Slot slot;
        slot.alloc = alloc;
        slot.size = type.getNumElements() * (width / 8);
        if (auto a = alloc.getAlignment())
          slot.align = std::max<int64_t>(*a, 1);
        // Anything that hands the buffer, or a view of it, to the caller takes
        // ownership with it.
        SmallVector<Value> reachable = {alloc.getResult()};
        for (unsigned i = 0; i < reachable.size(); i++) {
          for (Operation *user : reachable[i].getUsers()) {
            if (llvm::isa<func::ReturnOp>(user)) {
              slot.size = 0; // escapes: leave it to the allocator
              return;
            }
            if (isView(user))
              reachable.push_back(user->getResult(0));
          }
        }
        slots.push_back(slot);
      });
      if (slots.empty())
        return;

      // Each buffer gets its own space. Letting buffers whose lives do not
      // overlap share it -- the arena would then be the peak rather than the
      // sum, 0.36 MB against 4.45 on MobileNetV2 -- was tried and **is not
      // safe on this board**: a model whose answer was exact became wrong from
      // the second inference on, and wrong differently from run to run, while
      // the same object against the runtime's own CPU implementation stayed
      // exact. So it is not the planning: the data flow is right and something
      // about reusing an address for two roles is not. Adding slack between the
      // slots (256 bytes, then a page) moved the symptom without fixing it,
      // which rules out a write past the end. Not shipped until it is
      // understood; see docs/pipeline.md.
      int64_t total = 0;
      for (Slot &slot : slots) {
        slot.offset = roundUp(total, slot.align);
        total = slot.offset + slot.size;
      }

      auto i8 = builder.getIntegerType(8);
      auto arenaTy = MemRefType::get({total}, i8);
      std::string name =
          ("__gemmlir_arena_" + func.getName() + "_" + Twine(counter++)).str();
      builder.setInsertionPointToStart(module.getBody());
      // A UnitAttr initial value is how memref.global spells `uninitialized`;
      // leaving it out would make the symbol an external declaration.
      builder.create<memref::GlobalOp>(
          func.getLoc(), builder.getStringAttr(name),
          builder.getStringAttr("private"), TypeAttr::get(arenaTy),
          /*initial_value=*/builder.getUnitAttr(), /*constant=*/UnitAttr(),
          builder.getI64IntegerAttr(64));

      for (Slot &slot : slots) {
        OpBuilder local(slot.alloc);
        Value arena =
            local.create<memref::GetGlobalOp>(slot.alloc.getLoc(), arenaTy, name);
        Value shift = local.create<arith::ConstantIndexOp>(slot.alloc.getLoc(),
                                                           slot.offset);
        Value view = local.create<memref::ViewOp>(
            slot.alloc.getLoc(), llvm::cast<MemRefType>(slot.alloc.getType()),
            arena, shift, ValueRange{});
        slot.alloc.getResult().replaceAllUsesWith(view);
        slot.alloc.erase();
      }

      // The arena is never handed back, so neither are its pieces.
      SmallVector<memref::DeallocOp> dead;
      func.walk([&](memref::DeallocOp dealloc) {
        if (dealloc.getMemref().getDefiningOp<memref::ViewOp>())
          dead.push_back(dealloc);
      });
      for (memref::DeallocOp dealloc : dead)
        dealloc.erase();
    });
  }
};

} // namespace

} // namespace mlir::gemmlir
