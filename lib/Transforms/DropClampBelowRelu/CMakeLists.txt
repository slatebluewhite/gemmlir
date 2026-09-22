add_mlir_library(MLIRGemmlirDropClampBelowRelu
  DropClampBelowReluPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMathDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
