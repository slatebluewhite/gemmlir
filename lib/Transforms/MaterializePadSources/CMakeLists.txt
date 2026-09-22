add_mlir_library(MLIRGemmlirMaterializePadSources
  MaterializePadSourcesPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
