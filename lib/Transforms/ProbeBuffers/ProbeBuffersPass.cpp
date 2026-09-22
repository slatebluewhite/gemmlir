//===- ProbeBuffersPass.cpp --------------------------------------*- C++ -*-===//
//
// A checksum after every write, so two builds can be compared step by step.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_PROBEBUFFERS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The allocation a memref names, through the views that stay inside it.
static Value baseOf(Value v, unsigned depth = 0) {
  if (depth > 8)
    return nullptr;
  Operation *def = v.getDefiningOp();
  if (!def)
    return nullptr;
  if (llvm::isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp,
                memref::ViewOp>(def))
    return v;
  if (llvm::isa<memref::SubViewOp, memref::CastOp, memref::CollapseShapeOp,
                memref::ExpandShapeOp, memref::ReinterpretCastOp>(def))
    return baseOf(def->getOperand(0), depth + 1);
  return nullptr;
}

/// **A debugging pass, never in the pipeline.** It puts a call to
/// `gemmlir_probe(pointer, bytes, id)` after every operation that writes a
/// buffer, so two builds that should agree can be compared one write at a time
/// instead of only at the model's output.
///
/// It is here because the check that has caught everything else -- the same
/// object against the accelerator runtime and against the CPU one -- cannot see
/// a difference in the **generated host code**, since both runtimes share it.
/// Comparing two builds byte for byte says *that* they differ; this says
/// *where*.
///
/// The probe is on the whole allocation the write lands in, not the view: a
/// padded buffer written through a subview is exactly the case worth watching,
/// and its border is part of what the next operation reads.
class ProbeBuffers : public impl::ProbeBuffersBase<ProbeBuffers> {
public:
  using impl::ProbeBuffersBase<ProbeBuffers>::ProbeBuffersBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    auto i64 = IntegerType::get(ctx, 64);

    auto module = func->getParentOfType<ModuleOp>();
    if (!module)
      return;
    OpBuilder atTop(module.getBody(), module.getBody()->begin());
    StringRef probeName = "gemmlir_probe";
    if (!module.lookupSymbol(probeName)) {
      auto fn = atTop.create<func::FuncOp>(
          func.getLoc(), probeName,
          FunctionType::get(ctx, {i64, i64, i64}, {}));
      fn.setPrivate();
    }

    SmallVector<std::pair<Operation *, Value>> sites;
    func.walk([&](Operation *op) {
      Value written;
      if (auto dps = llvm::dyn_cast<DestinationStyleOpInterface>(op)) {
        if (dps.getNumDpsInits() == 1 &&
            llvm::isa<MemRefType>(dps.getDpsInits()[0].getType()))
          written = dps.getDpsInits()[0];
      } else if (auto copy = llvm::dyn_cast<memref::CopyOp>(op)) {
        written = copy.getTarget();
      }
      if (written)
        sites.push_back({op, written});
    });

    int64_t id = 0;
    for (auto [op, written] : sites) {
      Value base = baseOf(written);
      if (!base)
        continue;
      auto ty = llvm::dyn_cast<MemRefType>(base.getType());
      if (!ty || !ty.hasStaticShape() || !ty.getElementType().isIntOrFloat())
        continue;
      int64_t bytes =
          ty.getNumElements() * (ty.getElementType().getIntOrFloatBitWidth() / 8);

      OpBuilder b(op->getContext());
      b.setInsertionPointAfter(op);
      Location loc = op->getLoc();
      Value asIndex =
          b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, base);
      Value ptr = b.create<arith::IndexCastOp>(loc, i64, asIndex);
      Value n = b.create<arith::ConstantIntOp>(loc, bytes, 64);
      Value tag = b.create<arith::ConstantIntOp>(loc, id++, 64);
      b.create<func::CallOp>(loc, probeName, TypeRange{},
                             ValueRange{ptr, n, tag});
    }
  }
};

} // namespace

} // namespace mlir::gemmlir
