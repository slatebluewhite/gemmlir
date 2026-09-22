add_mlir_library(MLIRGemmlirPlanStaticBuffers
  PlanStaticBuffersPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRGemmlir
)
