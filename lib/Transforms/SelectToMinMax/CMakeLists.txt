add_mlir_library(MLIRGemmlirSelectToMinMax
  SelectToMinMaxPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
