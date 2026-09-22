add_mlir_library(MLIRGemmlirPointwiseConvToMatmul
  PointwiseConvToMatmulPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
