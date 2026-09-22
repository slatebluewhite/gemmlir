add_mlir_library(MLIRGemmlirQuantizeUnfoldableTails
  QuantizeUnfoldableTailsPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRQuantDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
