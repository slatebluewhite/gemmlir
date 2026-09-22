/* `--select-to-minmax`'s `ClampBeforeConversion` moves a quantization tail's
 * integer clamp into the float, because this board's Rocket is plain rv64gc and
 * `arith.maxsi`/`arith.minsi` are two data-dependent branches there.
 *
 * The claim is that it is **exact**: clamping commutes with rounding when the
 * bounds are integers, so `clamp(round(x))` and `round(clamp(x))` are the same
 * integer. The argument is short -- rounding is monotone and fixes integers --
 * and this checks it rather than trusting it twice, over **every** f32 bit
 * pattern in the range where the two could differ.
 *
 *   cc -O2 -o /tmp/clamp_check scripts/clamp_check.c -lm && /tmp/clamp_check
 *   checked 2267807744 values in (-300, 300), 0 mismatches
 *
 * Outside that range both saturate, and on a NaN or an infinity `fptosi` was
 * poison in the original, so there is nothing to preserve. */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
static int old_way(float x) {
  float r = nearbyintf(x);           /* math.roundeven */
  int32_t i = (int32_t)r;            /* arith.fptosi   */
  if (i < -128) i = -128;
  if (i > 127) i = 127;
  return i;
}
static int new_way(float x) {
  float c = fmaxf(x, -128.0f);
  c = fminf(c, 127.0f);
  return (int32_t)nearbyintf(c);
}
int main(void) {
  long bad = 0, n = 0;
  /* every representable f32 between -300 and 300, by bit pattern */
  for (uint32_t u = 0; u < 0xFFFFFFFFu; u += 1) {
    float x; memcpy(&x, &u, 4);
    if (!(x > -300.0f && x < 300.0f)) continue;
    n++;
    if (old_way(x) != new_way(x)) {
      if (bad < 5) printf("  MISMATCH x=%.9g old=%d new=%d\n", x, old_way(x), new_way(x));
      bad++;
    }
  }
  printf("checked %ld values in (-300, 300), %ld mismatches\n", n, bad);
  return bad != 0;
}
