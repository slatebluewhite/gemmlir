add_mlir_dialect(GemmlirOps gemmlir)
add_mlir_doc(GemmlirDialect GemmlirDialect Gemmlir/ -gen-dialect-doc)
add_mlir_doc(GemmlirOps GemmlirOps Gemmlir/ -gen-op-doc)

# The dataflow enum and its dialect attribute live in GemmlirOps.td; add_mlir_dialect
# only emits ops, types and the dialect itself, so drive the enum/attrdef backends here.
set(LLVM_TARGET_DEFINITIONS GemmlirOps.td)
mlir_tablegen(GemmlirEnums.h.inc -gen-enum-decls)
mlir_tablegen(GemmlirEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(GemmlirAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=gemmlir)
mlir_tablegen(GemmlirAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=gemmlir)
add_public_tablegen_target(MLIRGemmlirAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS GemmlirPasses.td)
mlir_tablegen(GemmlirPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRGemmlirPassesIncGen)
