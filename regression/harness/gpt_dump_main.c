/* A decoder-only transformer takes token ids, not an image. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/mman.h>
#include DATA_HEADER
float *forward(int64_t *idx);
enum { NT = sizeof torch_idx / sizeof *torch_idx,
       NY = sizeof torch_y / sizeof *torch_y };
static int64_t tb[NT];
int main(int argc, char **argv){
  setvbuf(stdout, NULL, _IOLBF, 0);
  mlockall(MCL_CURRENT|MCL_FUTURE);
  for (int i = 0; i < NT; i++) tb[i] = torch_idx[i];
  float *o = forward(tb);
  double s=0,n=0; for(int i=0;i<NY;i++){double d=o[i]-torch_y[i]; s+=d*d; n+=(double)torch_y[i]*torch_y[i];}
  printf("  relative L2 %.4f\n", sqrt(s/n));
  if (argc > 1) { FILE *f = fopen(argv[1], "wb"); if(!f){perror("fopen");return 1;} fwrite(o, sizeof *o, NY, f); fclose(f); }
  return 0; }
