add_mlir_dialect_library(MLIRGemmlir
        GemmlirTypes.cpp
        GemmlirDialect.cpp
        GemmlirOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Gemmlir

        DEPENDS
        MLIRGemmlirOpsIncGen
        MLIRGemmlirAttrsIncGen
        MLIRGemmlirPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRMemRefDialect
        MLIRTransforms
        )
