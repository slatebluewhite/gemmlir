add_mlir_library(MLIRGemmlirConvNchwToNhwc
  ConvNchwToNhwcPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
