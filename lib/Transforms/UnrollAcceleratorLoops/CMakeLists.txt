add_mlir_library(MLIRGemmlirUnrollAcceleratorLoops
  UnrollAcceleratorLoopsPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRTransforms
  MLIRGemmlir
)
