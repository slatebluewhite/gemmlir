add_mlir_library(MLIRGemmlirDepthwiseAsBlockDiagonal
  DepthwiseAsBlockDiagonalPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRGemmlir
)
