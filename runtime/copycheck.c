/* memrefCopy against a naive element-by-element reference, over many ranks,
   shapes, strides, element sizes and base alignments -- the run-collapsing,
   word, byte and memcpy paths all get exercised, and so does every way the
   odometer can carry. A runtime change has no IR to put a lit test on; this
   is the check that stands in for one, and it runs on the board. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

struct gemmlir_unranked_memref { int64_t rank; void *descriptor; };
void memrefCopy(int64_t elemSize, struct gemmlir_unranked_memref *srcArg,
                struct gemmlir_unranked_memref *dstArg);

enum { MAXRANK = 5, POOL = 1 << 16 };
static uint8_t srcBuf[POOL], dstBuf[POOL], refBuf[POOL];

static uint32_t seed = 0x2026u;
static uint32_t rnd(void){ seed = seed * 1664525u + 1013904223u; return seed >> 8; }

/* {allocated, aligned, offset, sizes..., strides...} */
static int64_t srcDesc[3 + 2 * MAXRANK], dstDesc[3 + 2 * MAXRANK];

static void naive(int64_t rank, const int64_t *sizes, const int64_t *ss,
                  const int64_t *ds, const uint8_t *s, uint8_t *d,
                  int64_t elemSize) {
  int64_t idx[MAXRANK] = {0};
  for (;;) {
    int64_t so = 0, doff = 0;
    for (int64_t k = 0; k < rank; k++) { so += idx[k] * ss[k]; doff += idx[k] * ds[k]; }
    memcpy(d + doff * elemSize, s + so * elemSize, (size_t)elemSize);
    int64_t axis = rank - 1;
    for (; axis >= 0; axis--) {
      if (++idx[axis] != sizes[axis]) break;
      idx[axis] = 0;
    }
    if (axis < 0) return;
  }
}

int main(void){
  int bad = 0, cases = 0;
  for (int trial = 0; trial < 20000; trial++) {
    int64_t rank = 1 + rnd() % MAXRANK;
    int64_t elemSize = (int64_t)1 << (rnd() % 4);          /* 1, 2, 4, 8 */
    int64_t sizes[MAXRANK], ss[MAXRANK], ds[MAXRANK];
    for (int64_t k = 0; k < rank; k++) sizes[k] = 1 + rnd() % 5;
    /* strides built innermost-out, with an optional gap so some dimensions
       are packed and some are not -- that is what run collapsing turns on. */
    int64_t sa = 1, da = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
      ss[k] = sa; ds[k] = da;
      sa *= sizes[k] + (rnd() % 3 == 0 ? 1 + rnd() % 2 : 0);
      da *= sizes[k] + (rnd() % 3 == 0 ? 1 + rnd() % 2 : 0);
    }
    int64_t srcOff = rnd() % 8, dstOff = rnd() % 8;        /* breaks alignment */
    if ((sa + srcOff) * elemSize >= POOL || (da + dstOff) * elemSize >= POOL) continue;
    cases++;
    for (int i = 0; i < POOL; i++) srcBuf[i] = (uint8_t)rnd();
    memset(dstBuf, 0xA5, POOL); memcpy(refBuf, dstBuf, POOL);

    srcDesc[0] = srcDesc[1] = (int64_t)(intptr_t)srcBuf; srcDesc[2] = srcOff;
    dstDesc[0] = dstDesc[1] = (int64_t)(intptr_t)dstBuf; dstDesc[2] = dstOff;
    for (int64_t k = 0; k < rank; k++) {
      srcDesc[3 + k] = sizes[k]; srcDesc[3 + rank + k] = ss[k];
      dstDesc[3 + k] = sizes[k]; dstDesc[3 + rank + k] = ds[k];
    }
    struct gemmlir_unranked_memref s = {rank, srcDesc}, d = {rank, dstDesc};
    memrefCopy(elemSize, &s, &d);
    naive(rank, sizes, ss, ds, srcBuf + srcOff * elemSize,
          refBuf + dstOff * elemSize, elemSize);
    if (memcmp(dstBuf, refBuf, POOL)) {
      if (bad < 5) {
        printf("  MISMATCH rank %ld elem %ld sizes", (long)rank, (long)elemSize);
        for (int64_t k = 0; k < rank; k++) printf(" %ld", (long)sizes[k]);
        printf("  srcStrides");
        for (int64_t k = 0; k < rank; k++) printf(" %ld", (long)ss[k]);
        printf("  dstStrides");
        for (int64_t k = 0; k < rank; k++) printf(" %ld", (long)ds[k]);
        printf("  offsets %ld %ld\n", (long)srcOff, (long)dstOff);
      }
      bad++;
    }
  }
  printf("memrefCopy: %d cases, %d disagree with the reference\n", cases, bad);
  return bad != 0;
}
