//===- LegalizeBarePtrReturnsPass.cpp ---------------------*- C++ -*-===//
//
// Makes a returned memref's allocated and aligned pointers coincide, so that
// the single pointer the bare-pointer calling convention hands back actually
// points at the data.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_LEGALIZEBAREPTRRETURNS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

class LegalizeBarePtrReturns
    : public impl::LegalizeBarePtrReturnsBase<LegalizeBarePtrReturns> {
public:
  using impl::LegalizeBarePtrReturnsBase<
      LegalizeBarePtrReturns>::LegalizeBarePtrReturnsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() final {
    getOperation().walk([](func::ReturnOp ret) {
      for (Value v : ret.getOperands()) {
        if (!llvm::isa<MemRefType>(v.getType()))
          continue;
        // `memref.alloc` with an alignment rounds the malloc result up and keeps
        // both pointers in the descriptor, but --convert-func-to-llvm returns
        // the *allocated* one (FuncToLLVM.cpp calls memrefDesc.allocatedPtr(),
        // its own comment notwithstanding). The caller then reads up to
        // `alignment - 1` bytes before the data. Dropping the attribute makes
        // the two pointers the same value, which is what the caller can use.
        //
        // The allocation is not always the returned value itself: a reshape
        // between them carries both pointers and the offset through unchanged,
        // so the returned pointer is still the allocation's. Missing that is
        // not a small error -- the whole result reads from the wrong address,
        // which is how a quantized convolution block came back at a relative
        // L2 of 0.94 against its own f32 reference.
        if (auto alloc = allocationBehind(v))
          if (alloc.getAlignment())
            alloc.removeAlignmentAttr();
      }
    });
  }

private:
  /// The allocation a returned memref actually points into, looking through
  /// operations that only reinterpret its shape and start at the same address.
  ///
  /// `--expand-strided-metadata` has already run by this point, so a reshape
  /// appears as a `memref.reinterpret_cast`; that one is followed only when its
  /// offset is a static zero, since an offset is not something dropping an
  /// alignment could fix. A `memref.subview` is never followed, for the same
  /// reason.
  static memref::AllocOp allocationBehind(Value v) {
    while (Operation *def = v.getDefiningOp()) {
      if (auto alloc = llvm::dyn_cast<memref::AllocOp>(def))
        return alloc;
      if (llvm::isa<memref::ExpandShapeOp, memref::CollapseShapeOp,
                    memref::CastOp>(def)) {
        v = def->getOperand(0);
        continue;
      }
      if (auto cast = llvm::dyn_cast<memref::ReinterpretCastOp>(def)) {
        SmallVector<OpFoldResult> offsets = cast.getMixedOffsets();
        if (offsets.size() != 1 || !isConstantIntValue(offsets[0], 0))
          return {};
        v = cast.getSource();
        continue;
      }
      return {};
    }
    return {};
  }
};

} // namespace

} // namespace mlir::gemmlir
