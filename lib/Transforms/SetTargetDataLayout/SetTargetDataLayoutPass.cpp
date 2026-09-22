//===- SetTargetDataLayoutPass.cpp -------------------------------*- C++ -*-===//
//
// Tell the translation which machine this is for.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_SETTARGETDATALAYOUT
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// Stamps `llvm.data_layout` and `llvm.target_triple` on the module.
///
/// Nothing set them, so `mlir-translate` handed `llc` a module with **no**
/// `target datalayout` line -- and LLVM's default layout says an `i64` is
/// aligned to **four** bytes. Every load and store of one is then printed
/// `align 4`, and the RISC-V backend, which must honour that, splits it:
///
/// ```
/// lwu  a5,-128(s1)     # instead of  ld a5,-128(s1)
/// lw   a1,-124(s1)
/// slli a1,a1,0x20
/// or   a1,a1,a5
/// ```
///
/// Four instructions where one would do, on every access -- and the same on the
/// way out, with an extra `srli` to split the value. `--pack-int8-max-pool`
/// exists to put eight channels in one `i64`, so it was paying that on every
/// load of every window: PC sampling put the SWAR pool at 5.7% of GoogLeNet,
/// the largest single block in `forward`, and half of it was this.
///
/// The claim has to be true, and it is: `--plan-static-buffers` starts every
/// slot at a multiple of 64, the arena itself is `align 64`, and the `i64` view
/// the packing pass builds sits at offset zero of one of those slots. A
/// `memref.alloc` that never reached the arena is aligned by the allocator to
/// at least the element's own size.
///
/// **Only the translation reads it.** `DataLayoutAnalysis`, which is what the
/// conversions consult for their own decisions, looks at `dlti.dl_spec`; this
/// attribute changes nothing before `mlir-translate`, which is why it can sit
/// at the end of the pipeline and touch nothing else.
///
/// | | ms | |
/// |---|---|---|
/// | `shufflenet_v2_x0_5` | 61.93 -> **47.31** | -23.6% |
/// | `googlenet`          | 409.05 -> **389.10** | -4.9% |
/// | `squeezenet1_1`      | 23.95 -> **23.19** | -3.2% |
/// | the set              | 2739.30 -> **2703.25** | -1.3% |
///
/// Nine of the other ten models are unchanged to within 0.4%, and every one of
/// the thirteen is byte for byte against both the CPU reference and the
/// previous build.
///
/// Two different splits were being paid. GoogLeNet and SqueezeNet had the `i64`
/// one above -- `lwu`+`lw`+`slli`+`or` collapses to one `ld`, and
/// `sw`+`sw`+`srli` to one `sd`. ShuffleNet and the ViT had a **byte-wise**
/// one: 29 of ShuffleNet's `lbu` disappear outright, because the vector copies
/// the channel shuffle is made of were being taken a byte at a time. That one
/// is worth five times more than the pool it was found through.
class SetTargetDataLayout
    : public impl::SetTargetDataLayoutBase<SetTargetDataLayout> {
public:
  using impl::SetTargetDataLayoutBase<
      SetTargetDataLayout>::SetTargetDataLayoutBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    Builder b(&getContext());
    if (!dataLayout.empty())
      module->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                      b.getStringAttr(dataLayout));
    if (!targetTriple.empty())
      module->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                      b.getStringAttr(targetTriple));
  }
};

} // namespace

} // namespace mlir::gemmlir
