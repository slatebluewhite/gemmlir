add_mlir_library(MLIRGemmlirApproximateExp
  ApproximateExpPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMathDialect
  MLIRMathTransforms
  MLIRVectorDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
