add_mlir_library(MLIRGemmlirSeparateMaxPool
  SeparateMaxPoolPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRGemmlir
)
