add_mlir_library(MLIRGemmlirRoundQuantizedCasts
  RoundQuantizedCastsPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRTransforms
  MLIRGemmlir
)
