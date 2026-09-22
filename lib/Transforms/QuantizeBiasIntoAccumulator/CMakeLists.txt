add_mlir_library(MLIRGemmlirQuantizeBiasIntoAccumulator
  QuantizeBiasIntoAccumulatorPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
