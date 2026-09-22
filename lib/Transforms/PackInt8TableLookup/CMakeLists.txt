add_mlir_library(MLIRGemmlirPackInt8TableLookup
  PackInt8TableLookupPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
  MLIRGemmlir
)
