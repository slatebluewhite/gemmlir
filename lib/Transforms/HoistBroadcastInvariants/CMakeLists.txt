add_mlir_library(MLIRGemmlirHoistBroadcastInvariants
  HoistBroadcastInvariantsPass.cpp

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
