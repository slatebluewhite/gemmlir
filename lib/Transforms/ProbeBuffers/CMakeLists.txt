add_mlir_library(MLIRGemmlirProbeBuffers
  ProbeBuffersPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
