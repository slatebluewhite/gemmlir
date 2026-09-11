add_mlir_library(MLIRGemmlirToLLVM
  GemmlirToLLVMPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRGemmlir
)

