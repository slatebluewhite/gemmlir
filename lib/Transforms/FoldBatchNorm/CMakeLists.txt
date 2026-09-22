add_mlir_library(MLIRGemmlirFoldBatchNorm
  FoldBatchNormPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransforms
  MLIRGemmlir
)
