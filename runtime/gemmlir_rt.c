/*
 * gemmlir runtime shim.
 *
 * The compiler lowers gemmlir.matmul_i8 to an external call
 *     void tiled_matmul_auto(size_t, size_t, size_t, const elem_t*, const elem_t*,
 *                            const void*, void*, size_t x4, scale_t, scale_t,
 *                            scale_acc_t, int, acc_scale_t, acc_scale_t,
 *                            bool x5, uint8_t, enum tiled_matmul_type_t)
 * which is defined `static` in gemmini.h. Defining EXPOSE_TOP_LEVEL_FNS turns those definitions into ordinary
 * external functions, so this single translation unit provides the symbol.
 *
 * Compile with your RISC-V toolchain and link it together with the object produced
 * by scripts/compile.sh:
 *     riscv64-unknown-linux-gnu-gcc -O2 -march=rv64gc -mabi=lp64d -c gemmlir_rt.c \
 *         -I<gemmini-rocc-tests> -include <your gemmini_params.h>   # -include: optional override
 */
#define EXPOSE_TOP_LEVEL_FNS
#include "include/gemmini.h"

/* The compiler hard-codes the argument types it emits. Catch a hardware config
 * whose generated gemmini_params.h disagrees, instead of silently passing bits
 * of the wrong type. */
#define GEMMLIR_IS_TYPE(T, want) _Generic((T)0, want: 1, default: 0)
_Static_assert(sizeof(size_t) == 8,
               "gemmlir emits dims/strides as i64; build for a 64-bit RISC-V target");
_Static_assert(GEMMLIR_IS_TYPE(elem_t, int8_t),
               "gemmlir emits int8 inputs; this gemmini_params.h uses a different elem_t");
_Static_assert(GEMMLIR_IS_TYPE(acc_t, int32_t),
               "gemmlir emits int32 accumulators; this gemmini_params.h uses a different acc_t");
_Static_assert(GEMMLIR_IS_TYPE(scale_t, float),
               "gemmlir emits A/B scale factors as f32; this gemmini_params.h uses a non-float scale_t");
_Static_assert(GEMMLIR_IS_TYPE(scale_acc_t, int32_t),
               "gemmlir emits D_scale_factor as i32; this gemmini_params.h uses a different scale_acc_t "
               "(e.g. float when accumulator mvin scaling is enabled). Adjust GemmlirToLLVMPass.cpp or the config.");
_Static_assert(GEMMLIR_IS_TYPE(acc_scale_t, float),
               "gemmlir emits scale/bert_scale as f32; this gemmini_params.h uses a non-float acc_scale_t");
