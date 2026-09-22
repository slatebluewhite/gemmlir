add_mlir_library(MLIRGemmlirRaiseSpatialSumToPool
  RaiseSpatialSumToPoolPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
