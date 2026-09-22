cmake_minimum_required(VERSION 3.20.0)
project(gemmlir LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

option(GEMMLIR_BUILD_RUNTIME "Build the Gemmini runtime shim (needs a RISC-V C compiler)" ON)
option(GEMMLIR_NORMALIZATIONS
       "Build the runtime with Gemmini's normalization unit (LayerNorm, softmax). \
Needs a bitstream whose Gemmini was built with norms = true; without one the RTL \
asserts. The two defines it adds are exactly what GemminiConfigs.scala emits for \
such a config -- the header shipped in third_party predates the norm-enabled \
board, so it does not carry them." OFF)

set(GEMMLIR_GEMMINI_PARAMS "" CACHE FILEPATH
    "gemmini_params.h generated for your hardware config (default: the one shipped in third_party/gemmini-rocc-tests)")
set(GEMMLIR_RISCV_CC "" CACHE FILEPATH "RISC-V C compiler used for the runtime shim (default: riscv64-unknown-elf-gcc on PATH)")

# ---------------------------------------------------------------------------
# LLVM / MLIR
# Pass -DMLIR_DIR=<llvm-build-or-install>/lib/cmake/mlir. See README for the LLVM
# commit and configuration this project is tested against.
# ---------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")
if(LLVM_VERSION_MAJOR LESS 22)
  message(FATAL_ERROR
    "gemmlir requires LLVM/MLIR >= 22 (found ${LLVM_PACKAGE_VERSION}). "
    "The pass implementation relies on opaque pointers, TableGen pass definitions and "
    "the MLIRRegisterAll* libraries. See README for building a matching LLVM.")
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(GEMMLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(GEMMLIR_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${GEMMLIR_SOURCE_DIR}/include)
include_directories(${GEMMLIR_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(gemmlir-opt)
add_subdirectory(test)
if(GEMMLIR_BUILD_RUNTIME)
  add_subdirectory(runtime)
endif()
