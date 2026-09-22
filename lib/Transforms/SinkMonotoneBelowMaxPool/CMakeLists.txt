add_mlir_library(MLIRGemmlirSinkMonotoneBelowMaxPool
  SinkMonotoneBelowMaxPoolPass.cpp

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
