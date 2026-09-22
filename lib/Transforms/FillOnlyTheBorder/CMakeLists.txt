add_mlir_library(MLIRGemmlirFillOnlyTheBorder
  FillOnlyTheBorderPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
