add_mlir_library(MLIRGemmlirShareBranchQuantization
  ShareBranchQuantizationPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRQuantDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
