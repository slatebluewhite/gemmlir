/*
 * Host-side driver for examples/conv_i8.mlir.
 *
 *   LLVM_BIN=<llvm-build>/bin ./scripts/compile.sh examples/conv_i8.mlir -o conv.o
 *   CFLAGS="-O2 -march=rv64gc -mabi=lp64d -static"
 *   riscv64-unknown-linux-gnu-gcc $CFLAGS -Ithird_party/gemmini-rocc-tests -c runtime/gemmlir_rt.c -o gemmlir_rt.o
 *   riscv64-unknown-linux-gnu-gcc $CFLAGS examples/conv_main.c conv.o gemmlir_rt.o -o conv
 *   ./conv          # on the board; prints PASS
 *
 * The reference is gemmini.h's own CPU implementation of the same three calls,
 * so this checks the compiler rather than the arithmetic.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "include/gemmini.h"

#define N 1
#define H 16
#define W 16
#define C 16
#define K 3
#define OH 14
#define OW 14
#define PH 7
#define PW 7

/* Bare-pointer calling convention: one pointer per memref. */
void conv_block(int8_t *in, int8_t *flt1, int32_t *bias1, int8_t *flt2,
                int8_t *window, int8_t *tmp1, int8_t *tmp2, int8_t *out);

static elem_t input[N * H * W * C] row_align(1);
static elem_t filter1[K * K * C * C] row_align(1);
static elem_t filter2[K * K * C * C] row_align(1);
static acc_t  bias1[C] row_align_acc(1);
static elem_t window[2 * 2] row_align(1);

/* The accelerator writes these, so they are allocated once and reused. */
static elem_t tmp1[N * PH * PW * C] row_align(1);
static elem_t tmp2[N * PH * PW * C] row_align(1);
static elem_t out[N * PH * PW * C] row_align(1);

static elem_t ref1[N * PH * PW * C] row_align(1);
static elem_t ref2[N * PH * PW * C] row_align(1);
static elem_t ref[N * PH * PW * C] row_align(1);

int main(void) {
  /* Gemmini's DMA does not fault pages in; see examples/main.c. */
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 2;
  }

  for (int i = 0; i < N * H * W * C; i++)
    input[i] = (elem_t)((i * 7) % 13 - 6);
  for (int i = 0; i < K * K * C * C; i++) {
    filter1[i] = (elem_t)((i * 5) % 11 - 5);
    filter2[i] = (elem_t)((i * 3) % 7 - 3);
  }
  for (int i = 0; i < C; i++)
    bias1[i] = (acc_t)(i * 37 - 300);

  gemmini_flush(0);
  tiled_conv_auto(N, H, W, C, C, OH, OW, 1, 1, 1, 0, K,
                  false, false, false, false, false,
                  input, filter1, bias1, ref1, RELU, 0.025f, 2, 2, 0, CPU);
  tiled_conv_auto(N, H, W, C, C, OH, OW, 1, 1, 1, 0, K,
                  false, false, false, false, false,
                  input, filter2, NULL, ref2, NO_ACTIVATION, 0.025f, 2, 2, 0, CPU);
  tiled_resadd_auto(PH, PW * C, 1.0f, 1.0f, 1.0f, ref1, ref2, ref, false, CPU);

  /* Repeated, because a fault here has shown up as every *other* call being
     wrong rather than all of them. */
  long mismatches = 0;
  for (int run = 0; run < 4; run++) {
    memset(out, 0, sizeof out);
    conv_block(input, filter1, bias1, filter2, window, tmp1, tmp2, out);
    for (int i = 0; i < N * PH * PW * C; i++)
      if (out[i] != ref[i] && mismatches++ < 5)
        printf("run %d: out[%d] = %d, expected %d\n", run, i, out[i], ref[i]);
  }

  printf(mismatches ? "FAIL (%ld mismatches)\n" : "PASS\n", mismatches);
  return mismatches ? 1 : 0;
}
