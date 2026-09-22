add_mlir_library(MLIRGemmlirUnrollReductionWindows
  UnrollReductionWindowsPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
