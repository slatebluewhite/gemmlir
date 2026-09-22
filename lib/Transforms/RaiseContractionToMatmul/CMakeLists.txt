add_mlir_library(MLIRGemmlirRaiseContractionToMatmul
  RaiseContractionToMatmulPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
