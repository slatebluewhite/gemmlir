add_mlir_library(MLIRGemmlirPromoteReductionAccumulator
  PromoteReductionAccumulatorPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
