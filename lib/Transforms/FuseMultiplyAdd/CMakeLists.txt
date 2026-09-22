add_mlir_library(MLIRGemmlirFuseMultiplyAdd
  FuseMultiplyAddPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
