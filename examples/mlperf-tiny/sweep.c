#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include DATA_HEADER
float *forward(float *x);
enum { NX = sizeof torch_x / sizeof *torch_x, NY = sizeof torch_y / sizeof *torch_y };
static float xb[NX];
int main(int argc, char **argv){
  mlockall(MCL_CURRENT|MCL_FUTURE);
  FILE *f = fopen(argc > 1 ? argv[1] : "/dev/null", "wb");
  if (!f){ printf("cannot open\n"); return 1; }
  for (int rep = 0; rep < 3; rep++)
    for (int k = 0; k < 3; k++){
      float s = 1.0f + 0.25f * k;
      for (int i = 0; i < NX; i++) xb[i] = torch_x[i] * s;
      float *o = forward(xb);
      fwrite(o, sizeof *o, NY, f);
      if (rep == 0 && k == 0){
        double e = 0, n = 0;
        for (int i = 0; i < NY; i++){ double d = o[i] - torch_y[i]; e += d*d; n += (double)torch_y[i]*torch_y[i]; }
        printf("%.4f", sqrt(e/n));
      }
    }
  fclose(f);
  return 0; }
