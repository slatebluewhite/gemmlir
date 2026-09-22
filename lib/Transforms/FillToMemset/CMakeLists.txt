add_mlir_library(MLIRGemmlirFillToMemset
  FillToMemsetPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
  MLIRGemmlir
)
