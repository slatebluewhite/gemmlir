get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        # MLIR >= 21 ships registerAllDialects/registerAllPasses as libraries.
        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
        MLIRRegisterAllExtensions
        MLIRGemmlir
        MLIRLinalgToGemmlir
        MLIRForceQuantizedMatmul
        MLIRGemmlirToLLVM
        )
add_llvm_executable(gemmlir-opt gemmlir-opt.cpp)
llvm_update_compile_flags(gemmlir-opt)
target_link_libraries(gemmlir-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(gemmlir-opt)
