add_mlir_library(MLIRGemmlirClampAsRangeCheck
  ClampAsRangeCheckPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTransforms
  MLIRGemmlir
)
