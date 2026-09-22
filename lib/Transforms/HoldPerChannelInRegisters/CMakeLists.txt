add_mlir_library(MLIRGemmlirHoldPerChannelInRegisters
  HoldPerChannelInRegistersPass.cpp

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
