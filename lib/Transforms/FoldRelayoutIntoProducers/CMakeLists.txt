add_mlir_library(MLIRGemmlirFoldRelayoutIntoProducers
  FoldRelayoutIntoProducersPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
