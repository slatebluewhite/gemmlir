/* Run the walkthrough's block and say whether it is right.
 *
 * Same gate the model set uses: this object linked against the accelerator
 * runtime and against gemmini.h's own CPU implementation, and the two outputs
 * compared byte for byte. relative L2 is against PyTorch's float answer.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include "11_data.h"
float *forward(float *x);
enum { NX = sizeof torch_x / sizeof *torch_x, NY = sizeof torch_y / sizeof *torch_y };
static float xb[NX];
int main(int argc, char **argv) {
  mlockall(MCL_CURRENT | MCL_FUTURE);
  memcpy(xb, torch_x, sizeof xb);
  float *o = forward(xb);
  double s = 0, n = 0;
  for (int i = 0; i < NY; i++) { double d = o[i] - torch_y[i]; s += d*d; n += (double)torch_y[i]*torch_y[i]; }
  printf("  %d inputs -> %d outputs   relative L2 %.4f\n", NX, NY, sqrt(s/n));
  if (argc > 1) { FILE *f = fopen(argv[1], "wb"); fwrite(o, sizeof *o, NY, f); fclose(f); }
  return 0;
}
