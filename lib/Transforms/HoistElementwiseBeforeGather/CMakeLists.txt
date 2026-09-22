add_mlir_library(MLIRGemmlirHoistElementwiseBeforeGather
  HoistElementwiseBeforeGatherPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMathDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRGemmlir
)
