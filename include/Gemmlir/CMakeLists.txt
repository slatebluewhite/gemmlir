add_mlir_dialect(GemmlirOps gemmlir)
add_mlir_doc(GemmlirDialect GemmlirDialect Gemmlir/ -gen-dialect-doc)
add_mlir_doc(GemmlirOps GemmlirOps Gemmlir/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS GemmlirPasses.td)
mlir_tablegen(GemmlirPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRGemmlirPassesIncGen)
