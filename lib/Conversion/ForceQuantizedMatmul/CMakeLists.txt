add_mlir_library(MLIRForceQuantizedMatmul
  ForceQuantizedMatmulPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRQuantDialect
  MLIRTensorDialect
  MLIRTransforms
)
