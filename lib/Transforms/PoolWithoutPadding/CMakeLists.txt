add_mlir_library(MLIRGemmlirPoolWithoutPadding
  PoolWithoutPaddingPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
