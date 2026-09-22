add_mlir_library(MLIRGemmlirFuseElementwiseAroundMatmul
  FuseElementwiseAroundMatmulPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
  MLIRGemmlirConvNchwToNhwc
)
