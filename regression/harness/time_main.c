#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include DATA_HEADER
float *forward(float *x);
#include "ticks.h"
enum { NX = sizeof torch_x / sizeof *torch_x, NY = sizeof torch_y / sizeof *torch_y };
static float xb[NX];
int main(void){
  mlockall(MCL_CURRENT|MCL_FUTURE);
  for (int r=0;r<3;r++){ memcpy(xb, torch_x, sizeof xb); float *o=forward(xb);
    double s=0,n=0; for(int i=0;i<NY;i++){double d=o[i]-torch_y[i]; s+=d*d; n+=(double)torch_y[i]*torch_y[i];}
    if (r==2) printf("  relative L2 %.4f\n", sqrt(s/n)); }
  memcpy(xb, torch_x, sizeof xb); forward(xb);
  uint64_t t0 = ticks();
  for (int r=0;r<REPS;r++){ memcpy(xb, torch_x, sizeof xb); forward(xb); }
  printf("  %8.2f ms/inference\n", (ticks()-t0)*tick_ms()/(double)REPS);
  return 0; }
