add_mlir_library(MLIRGemmlirFoldChannelShuffle
  FoldChannelShufflePass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
