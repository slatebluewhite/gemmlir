/* The harness bytecheck.sh links: one inference, the answer written out.
 *
 * One inference per process on purpose. A fault that only catches the first
 * issue of an accelerator configuration is invisible to a loop inside one
 * process -- the second inference reuses the same configuration and comes out
 * right -- so the sample has to be across processes. */
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include DATA_HEADER

float *forward(float *x);

enum { NX = sizeof torch_x / sizeof *torch_x,
       NY = sizeof torch_y / sizeof *torch_y };
static float xb[NX];

int main(int argc, char **argv) {
  mlockall(MCL_CURRENT | MCL_FUTURE);
  memcpy(xb, torch_x, sizeof xb);
  float *o = forward(xb);
  FILE *f = fopen(argc > 1 ? argv[1] : "/dev/shm/got.bin", "wb");
  if (!f) { perror("fopen"); return 1; }
  fwrite(o, sizeof *o, NY, f);
  fclose(f);
  return 0;
}
