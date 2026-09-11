add_mlir_library(MLIRLinalgToGemmlir
  LinalgToGemmlirPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRGemmlir
)

