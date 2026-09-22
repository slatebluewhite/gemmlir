add_mlir_library(MLIRGemmlirGatherToMemrefCopy
  GatherToMemrefCopyPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRGemmlir
)
