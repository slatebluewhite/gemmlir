add_mlir_library(MLIRGemmlirPlaceCacheFlushes
  PlaceCacheFlushesPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLinalgDialect
  MLIRTransforms
  MLIRGemmlir
)
