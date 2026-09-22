add_mlir_library(MLIRGemmlirSaturateConstantCasts
  SaturateConstantCastsPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
