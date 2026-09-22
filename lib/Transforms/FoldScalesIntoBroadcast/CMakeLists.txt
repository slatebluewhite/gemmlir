add_mlir_library(MLIRGemmlirFoldScalesIntoBroadcast
  FoldScalesIntoBroadcastPass.cpp

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
