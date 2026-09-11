# Builds runtime/gemmlir_rt.o with a RISC-V C compiler. Skipped (with a message) when
# no such compiler is available; the source can always be compiled by hand.
set(GEMMINI_ROCC_TESTS ${PROJECT_SOURCE_DIR}/third_party/gemmini-rocc-tests)
if(NOT EXISTS ${GEMMINI_ROCC_TESTS}/include/gemmini.h)
  message(WARNING "third_party/gemmini-rocc-tests is empty. Run: make update-submodule")
  return()
endif()

if(GEMMLIR_RISCV_CC)
  set(_riscv_cc ${GEMMLIR_RISCV_CC})
else()
  find_program(_riscv_cc NAMES riscv64-unknown-linux-gnu-gcc riscv64-unknown-elf-gcc)
endif()
if(NOT _riscv_cc)
  message(STATUS "No RISC-V C compiler found; skipping the gemmlir-rt target "
                 "(set -DGEMMLIR_RISCV_CC=... or add riscv64-unknown-linux-gnu-gcc to PATH)")
  return()
endif()
message(STATUS "Runtime shim compiler: ${_riscv_cc}")

set(_params_flags)
if(GEMMLIR_GEMMINI_PARAMS)
  # -include makes the user's header win over the submodule's default one.
  set(_params_flags -include ${GEMMLIR_GEMMINI_PARAMS})
  message(STATUS "gemmini_params.h override: ${GEMMLIR_GEMMINI_PARAMS}")
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gemmlir_rt.o
  COMMAND ${_riscv_cc} -O2 -march=rv64gc -mabi=lp64d
          -I${GEMMINI_ROCC_TESTS} ${_params_flags}
          -c ${CMAKE_CURRENT_SOURCE_DIR}/gemmlir_rt.c -o ${CMAKE_CURRENT_BINARY_DIR}/gemmlir_rt.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gemmlir_rt.c
  COMMENT "Compiling Gemmini runtime shim (RISC-V)")
add_custom_target(gemmlir-rt ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gemmlir_rt.o)
