add_mlir_library(MLIRGemmlirHoistInvariantReciprocal
  HoistInvariantReciprocalPass.cpp

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
