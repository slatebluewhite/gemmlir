add_mlir_library(MLIRGemmlirPackInt8MaxPool
  PackInt8MaxPoolPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
