add_mlir_library(MLIRGemmlirUnrollElementwiseLoops
  UnrollElementwiseLoopsPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
