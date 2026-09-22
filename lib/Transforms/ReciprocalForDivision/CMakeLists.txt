add_mlir_library(MLIRGemmlirReciprocalForDivision
  ReciprocalForDivisionPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
