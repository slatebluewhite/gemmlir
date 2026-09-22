/* Runs the block `from_torch.py` exported, and says how far it is from what
 * PyTorch computed in f32.
 *
 * Link it twice -- once against `gemmlir_rt.o`, once against
 * `gemmlir_rt_cpu.o` -- and the two binaries are the *same compiled object*
 * with the accelerator taken away. That is what the speedup is measured
 * against, and why the outputs can be required to match byte for byte.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include "shufflenet_data.h"

float *forward(float *x);

/* `rdtime` counts a 500 kHz timebase on this board: 2.0 us a tick. Read it
 * rather than assuming, because the bitstream's clock has changed before. */
static double tick_ms(void) { return 1000.0 / 500000.0; }
static uint64_t ticks(void) {
  uint64_t t;
  asm volatile("rdtime %0" : "=r"(t));
  return t;
}

enum { NX = sizeof torch_x / sizeof *torch_x,
       NY = sizeof torch_y / sizeof *torch_y,
       REPS = 10 };
static float xb[NX];

int main(void) {
  mlockall(MCL_CURRENT | MCL_FUTURE);
  memcpy(xb, torch_x, sizeof xb);
  float *y = forward(xb);                    /* warm: the first call to a
                                                buffer is the one the runtime
                                                has to touch itself */
  double err = 0, norm = 0;
  for (int i = 0; i < NY; i++) {
    double d = y[i] - torch_y[i];
    err += d * d;
    norm += (double)torch_y[i] * torch_y[i];
  }

  uint64_t t0 = ticks();
  for (int r = 0; r < REPS; r++) {
    memcpy(xb, torch_x, sizeof xb);
    forward(xb);
  }
  uint64_t t1 = ticks();

  printf("%8.2f ms/inference   relative L2 %.4f\n",
         (t1 - t0) * tick_ms() / REPS, sqrt(err / norm));
  return 0;
}
