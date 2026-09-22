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
 *
 * It runs four rounds over three different inputs -- see the comment in main().
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

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
  /* Gemmini's DMA works on physical addresses and does not fault pages in. On a
     Linux board an unpinned page therefore reads as whatever was there and the
     matmul comes back partly zero -- measured on a 62.5 MHz U280 Rocket SoC:
     without this call the check below reports ~31.5k of 32768 elements wrong,
     deterministically, and with it the same binary passes. Every Linux test in
     gemmini-rocc-tests opens the same way. Called unconditionally: this example
     targets Linux, and keying it on BAREMETAL invites defining that macro to 0
     and silently losing the call. */
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 2;
  }

  /* Four rounds, three different inputs, with the first repeated at the end.
     One round proves nothing about a board whose accelerator can read past the
     host's writes: the host's stores may still be in its L1 while Gemmini reads
     from the L2, and the stale data it reads is then the *previous* round's, so
     a driver that replays one input gets the right answer from the wrong data.
     Measured before `runtime/gemmlir_rt.c` walked the L1 on both sides of every
     call: three of six models came back wrong on every input after the first,
     and the repeat of the first input no longer matched itself. */
  static const int seeds[] = {0, 1, 2, 0};
  long mismatches = 0;
  for (unsigned r = 0; r < sizeof seeds / sizeof *seeds; r++) {
    int s = seeds[r];
    for (int i = 0; i < M; i++)
      for (int k = 0; k < K; k++)
        A[i][k] = (int8_t)((i * 7 + k * 3 + s * 31) % 13 - 6);
    for (int k = 0; k < K; k++)
      for (int n = 0; n < N; n++)
        B[k][n] = (int8_t)((k * 5 + n * 11 + s * 23) % 17 - 8);

    for (int i = 0; i < M; i++)
      for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++)
          acc += (int32_t)A[i][k] * (int32_t)B[k][n];
        ref[i][n] = acc;
      }

    /* `linalg.matmul` on memrefs means `C += A*B`, and the lowering honours
       that through the accelerator's D operand, so C is the running sum and
       has to start at zero. The single-round version of this driver got away
       without the memset because C is a zero-initialised static. */
    for (int i = 0; i < M; i++)
      for (int n = 0; n < N; n++)
        C[i][n] = 0;
    /* full_C = true in the lowering: C receives raw int32 accumulators. */
    matmul_example(&A[0][0], &B[0][0], &C[0][0]);

    long bad = 0;
    for (int i = 0; i < M; i++)
      for (int n = 0; n < N; n++)
        if (C[i][n] != ref[i][n] && bad++ < 5)
          printf("round %u (seed %d): C[%d][%d] = %d, expected %d\n", r, s, i, n,
                 C[i][n], ref[i][n]);
    if (bad)
      printf("round %u (seed %d): %ld mismatches\n", r, s, bad);
    mismatches += bad;
  }

  printf(mismatches ? "FAIL (%ld mismatches)\n" : "PASS\n", mismatches);
  return mismatches ? 1 : 0;
}
