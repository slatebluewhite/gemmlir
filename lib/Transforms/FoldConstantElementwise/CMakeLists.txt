add_mlir_library(MLIRGemmlirFoldConstantElementwise
  FoldConstantElementwisePass.cpp

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
