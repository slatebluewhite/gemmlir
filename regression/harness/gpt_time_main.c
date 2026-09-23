#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include DATA_HEADER
float *forward(int64_t *idx);
#include "ticks.h"
enum { NT = sizeof torch_idx / sizeof *torch_idx,
       NY = sizeof torch_y / sizeof *torch_y };
static int64_t tb[NT];
int main(void){
  mlockall(MCL_CURRENT|MCL_FUTURE);
  for (int i = 0; i < NT; i++) tb[i] = torch_idx[i];
  float *o = forward(tb);
  double s=0,n=0; for(int i=0;i<NY;i++){double d=o[i]-torch_y[i]; s+=d*d; n+=(double)torch_y[i]*torch_y[i];}
  printf("  relative L2 %.4f\n", sqrt(s/n));
  uint64_t t0 = ticks();
  for (int r=0;r<REPS;r++){ for (int i = 0; i < NT; i++) tb[i] = torch_idx[i]; forward(tb); }
  printf("  %8.2f ms/inference\n", (ticks()-t0)*tick_ms()/(double)REPS);
  return 0; }
