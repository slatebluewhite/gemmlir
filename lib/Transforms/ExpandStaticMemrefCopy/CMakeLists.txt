add_mlir_library(MLIRGemmlirExpandStaticMemrefCopy
  ExpandStaticMemrefCopyPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRVectorDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
