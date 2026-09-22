add_mlir_library(MLIRGemmlirRequantizeBeforePooling
  RequantizeBeforePoolingPass.cpp

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
