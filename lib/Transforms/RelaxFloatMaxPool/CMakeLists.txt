add_mlir_library(MLIRGemmlirRelaxFloatMaxPool
  RelaxFloatMaxPoolPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
