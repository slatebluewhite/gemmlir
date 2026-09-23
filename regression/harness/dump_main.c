#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/mman.h>
#include DATA_HEADER
float *forward(float *x);
enum { NX = sizeof torch_x / sizeof *torch_x, NY = sizeof torch_y / sizeof *torch_y };
static float xb[NX];
int main(int argc, char **argv){
  setvbuf(stdout, NULL, _IOLBF, 0);
  int lock = getenv("LOCK") != NULL;
  if (lock) printf("  mlockall %s\n", mlockall(MCL_CURRENT|MCL_FUTURE) ? "FAILED" : "ok");
  memcpy(xb, torch_x, sizeof xb);
  float *o = forward(xb);
  double s=0,n=0,m=0; for(int i=0;i<NY;i++){double d=o[i]-torch_y[i]; s+=d*d; n+=(double)torch_y[i]*torch_y[i]; if(fabs(o[i])>m) m=fabs(o[i]);}
  printf("  L2 %.4f   |o|max %.6f\n", sqrt(s/n), m);
  if (argc > 1) { FILE *f = fopen(argv[1], "wb"); if(!f){perror("fopen");return 1;} fwrite(o, sizeof *o, NY, f); fclose(f); }
  return 0; }
