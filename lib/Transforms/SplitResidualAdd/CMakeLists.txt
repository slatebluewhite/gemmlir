add_mlir_library(MLIRGemmlirSplitResidualAdd
  SplitResidualAddPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
