add_mlir_library(MLIRGemmlirCombineConstantScales
  CombineConstantScalesPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRTransforms
  MLIRGemmlir
)
