add_mlir_library(MLIRGemmlirFoldRequantizeIntoSliceMatmuls
  FoldRequantizeIntoSliceMatmulsPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRGemmlir
)
