/* What the board sees.
 *
 * The validation harness prints a relative L2 because it runs random weights on
 * random noise -- there is no "answer" to get right. This one runs torchvision's
 * own pretrained weights on real photographs, so there is: the class. It prints
 * the board's top five, and next to it what PyTorch's float32 model said, so the
 * cost of the whole int8 pipeline is visible as a difference in names and
 * percentages rather than a number nobody can picture.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/mman.h>
#include DATA_HEADER
#include "ticks.h"

float *forward(float *x);
static float xb[NPIX];

static void top5(const float *v, int *idx, float *prob) {
  double m = -1e30;
  for (int i = 0; i < NCLS; i++) if (v[i] > m) m = v[i];
  double s = 0;
  for (int i = 0; i < NCLS; i++) s += exp(v[i] - m);
  char taken[NCLS];
  memset(taken, 0, sizeof taken);
  for (int k = 0; k < 5; k++) {
    int best = -1;
    for (int i = 0; i < NCLS; i++)
      if (!taken[i] && (best < 0 || v[i] > v[best])) best = i;
    taken[best] = 1; idx[k] = best; prob[k] = (float)(exp(v[best] - m) / s);
  }
}

int main(void) {
  setvbuf(stdout, NULL, _IOLBF, 0);
  mlockall(MCL_CURRENT | MCL_FUTURE);
  printf("\n  %s, int8 on Gemmini -- %d classes, %d images\n\n", MODEL_NAME, NCLS, NIMG);

  /* One inference before the clock starts. The first one on this board costs
   * about forty times the rest -- the weights are moved in, the pages are
   * touched for the first time, and `tiled_conv_*_auto` searches its tiling
   * before the memo has anything in it. None of that is per image. */
  memcpy(xb, torch_x[0], sizeof xb);
  { uint64_t t0 = ticks(); forward(xb);
    printf("  first inference (weights moved in, tiling searched): %.0f ms\n\n",
           (ticks() - t0) * tick_ms()); }

  int agree = 0;
  double total = 0;
  for (int n = 0; n < NIMG; n++) {
    memcpy(xb, torch_x[n], sizeof xb);
    uint64_t t0 = ticks();
    float *o = forward(xb);
    double ms = (ticks() - t0) * tick_ms();
    total += ms;

    int bi[5], ti[5]; float bp[5], tp[5];
    top5(o, bi, bp);
    top5(torch_y[n], ti, tp);
    double s = 0, d = 0;
    for (int i = 0; i < NCLS; i++) { double e = o[i] - torch_y[n][i]; s += e * e; d += (double)torch_y[n][i] * torch_y[n][i]; }
    agree += (bi[0] == ti[0]);

    printf("  %-8s  %8.1f ms   relative L2 %.4f   %s\n",
           img_name[n], ms, sqrt(s / d),
           bi[0] == ti[0] ? "top-1 agrees with torch" : "TOP-1 DIFFERS");
    for (int k = 0; k < 5; k++)
      printf("      %d. %-24s %5.1f%%      torch: %-24s %5.1f%%\n",
             k + 1, class_name[bi[k]], 100.0 * bp[k], class_name[ti[k]], 100.0 * tp[k]);
    printf("\n");
  }
  printf("  top-1 agrees on %d of %d,  %.1f ms an image\n\n", agree, NIMG, total / NIMG);
  printf("DEMODONE\n");
  return 0;
}
