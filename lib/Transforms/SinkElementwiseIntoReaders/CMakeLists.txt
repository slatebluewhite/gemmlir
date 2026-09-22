add_mlir_library(MLIRGemmlirSinkElementwiseIntoReaders
  SinkElementwiseIntoReadersPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRGemmlir
)
