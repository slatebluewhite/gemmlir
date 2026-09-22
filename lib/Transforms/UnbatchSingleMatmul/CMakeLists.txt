add_mlir_library(MLIRGemmlirUnbatchSingleMatmul
  UnbatchSingleMatmulPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
