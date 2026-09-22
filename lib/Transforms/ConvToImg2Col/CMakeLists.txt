add_mlir_library(MLIRGemmlirConvToImg2Col
  ConvToImg2ColPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
