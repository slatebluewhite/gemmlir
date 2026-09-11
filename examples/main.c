/*
 * Host-side driver for examples/matmul_i8.mlir.
 *
 * matmul_example is produced by scripts/compile.sh from the MLIR file; its shapes
 * (128x128 x 128x256 -> 128x256) are fixed at compile time. The Gemmini runtime comes
 * from runtime/gemmlir_rt.o. Build for a Gemmini-enabled Rocket SoC running Linux:
 *
 *   LLVM_BIN=<llvm-build>/bin ./scripts/compile.sh examples/matmul_i8.mlir -o matmul.o
 *   CFLAGS="-O2 -march=rv64gc -mabi=lp64d -static"
 *   riscv64-unknown-linux-gnu-gcc $CFLAGS -Ithird_party/gemmini-rocc-tests -c runtime/gemmlir_rt.c -o gemmlir_rt.o
 *   riscv64-unknown-linux-gnu-gcc $CFLAGS examples/main.c matmul.o gemmlir_rt.o -o matmul
 *   ./matmul        # on the board; prints PASS
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define M 128
#define K 128
#define N 256

/* Bare-pointer calling convention: one pointer per memref, row-major. */
void matmul_example(int8_t *A, int8_t *B, int32_t *C);

static int8_t A[M][K];
static int8_t B[K][N];
static int32_t C[M][N];
static int32_t ref[M][N];

int main(void) {
  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++)
      A[i][k] = (int8_t)((i * 7 + k * 3) % 13 - 6);
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      B[k][n] = (int8_t)((k * 5 + n * 11) % 17 - 8);

  for (int i = 0; i < M; i++)
    for (int n = 0; n < N; n++) {
      int32_t acc = 0;
      for (int k = 0; k < K; k++)
        acc += (int32_t)A[i][k] * (int32_t)B[k][n];
      ref[i][n] = acc;
    }

  /* full_C = true in the lowering: C receives raw int32 accumulators, C is not read. */
  matmul_example(&A[0][0], &B[0][0], &C[0][0]);

  long mismatches = 0;
  for (int i = 0; i < M; i++)
    for (int n = 0; n < N; n++)
      if (C[i][n] != ref[i][n] && mismatches++ < 5)
        printf("C[%d][%d] = %d, expected %d\n", i, n, C[i][n], ref[i][n]);

  printf(mismatches ? "FAIL (%ld mismatches)\n" : "PASS\n", mismatches);
  return mismatches ? 1 : 0;
}
