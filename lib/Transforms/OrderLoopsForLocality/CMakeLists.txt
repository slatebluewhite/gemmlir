add_mlir_library(MLIRGemmlirOrderLoopsForLocality
  OrderLoopsForLocalityPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRGemmlir
)
