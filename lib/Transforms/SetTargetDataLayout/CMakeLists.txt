add_mlir_library(MLIRGemmlirSetTargetDataLayout
  SetTargetDataLayoutPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRTransforms
  MLIRGemmlir
)
