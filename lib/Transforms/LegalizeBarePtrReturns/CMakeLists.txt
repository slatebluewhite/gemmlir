add_mlir_library(MLIRGemmlirLegalizeBarePtrReturns
  LegalizeBarePtrReturnsPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRGemmlir
)
