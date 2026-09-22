/* Is a byte-wise SWAR max worth a pass?  One representative GoogLeNet pool:
 * 14x14x256 i8 NHWC padded input, 3x3 stride 1, 12x12x256 out.
 * The values are a relu's output, so every byte is in [0, 127] -- which is what
 * makes the cheap comparison correct. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ticks.h"

#define H 14
#define W 14
#define C 256
#define OH 12
#define OW 12
#define K 3

static int8_t in[H * W * C];
static int8_t out_scalar[OH * OW * C];
static int8_t out_swar[OH * OW * C];

static void pool_scalar(void) {
  for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
      for (int c = 0; c < C; c++) {
        int8_t m = -128;
        for (int kh = 0; kh < K; kh++)
          for (int kw = 0; kw < K; kw++) {
            int8_t v = in[((oh + kh) * W + (ow + kw)) * C + c];
            if (v > m) m = v;
          }
        out_scalar[(oh * OW + ow) * C + c] = m;
      }
}

/* Byte-wise unsigned max of two words whose bytes are all below 0x80.
 * `(a | HI) - b` cannot borrow out of a byte, and bit 7 of each byte of the
 * difference is 1 exactly when a_i >= b_i. */
#define HI  0x8080808080808080ull
#define LOW 0x0101010101010101ull
static inline uint64_t umax8(uint64_t a, uint64_t b) {
  uint64_t d = (a | HI) - b;
  uint64_t m = ((d >> 7) & LOW) * 0xFFull;   /* 0xFF per byte where a_i >= b_i */
  return b ^ ((a ^ b) & m);
}

/* The general signed version: every byte flipped into unsigned order once on
 * the way in and once on the way out, and a comparison that survives a byte
 * with its high bit set. */
static inline uint64_t umax8_general(uint64_t a, uint64_t b) {
  /* Write a byte as 0x80*ah + al.  `(a|HI) - (b & ~HI)` is 0x80 + al - bl per
   * byte, which cannot borrow out, so its bit 7 is `al >= bl`.  Unsigned
   * `a >= b` is then `ah & ~bh`, or `al >= bl` when the high bits agree. */
  uint64_t d = (a | HI) - (b & ~HI);
  uint64_t ge = HI & ((a & ~b) | (~(a ^ b) & d));
  uint64_t m = ((ge >> 7) & LOW) * 0xFFull;
  return b ^ ((a ^ b) & m);
}

static int8_t out_gen[OH * OW * C];

static void pool_general(void) {
  for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
      for (int c = 0; c < C; c += 8) {
        uint64_t m = 0;                       /* -128 flipped is 0x00 */
        for (int kh = 0; kh < K; kh++)
          for (int kw = 0; kw < K; kw++) {
            uint64_t w;
            memcpy(&w, &in[((oh + kh) * W + (ow + kw)) * C + c], 8);
            m = umax8_general(m, w ^ HI);
          }
        m ^= HI;
        memcpy(&out_gen[(oh * OW + ow) * C + c], &m, 8);
      }
}

static void pool_swar(void) {
  for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
      for (int c = 0; c < C; c += 8) {
        uint64_t m = 0;                       /* every byte is >= 0 */
        for (int kh = 0; kh < K; kh++)
          for (int kw = 0; kw < K; kw++) {
            uint64_t w;
            memcpy(&w, &in[((oh + kh) * W + (ow + kw)) * C + c], 8);
            m = umax8(m, w);
          }
        memcpy(&out_swar[(oh * OW + ow) * C + c], &m, 8);
      }
}

int main(void) {
  uint32_t s = 12345;
  for (int i = 0; i < H * W * C; i++) { s = s * 1103515245u + 12345u; in[i] = (s >> 16) & 0x7F; }
  /* every byte pair, both formulas, against the definition */
  int bad_c = 0, bad_g = 0;
  for (int x = 0; x < 256; x++)
    for (int y = 0; y < 256; y++) {
      uint64_t a = (uint64_t)(uint8_t)x * LOW, b = (uint64_t)(uint8_t)y * LOW;
      if ((uint8_t)(umax8_general(a ^ HI, b ^ HI) ^ HI) !=
          (uint8_t)((int8_t)x > (int8_t)y ? x : y))
        bad_g++;
      if (x < 128 && y < 128 && (uint8_t)umax8(a, b) != (uint8_t)(x > y ? x : y))
        bad_c++;
    }
  /* and mixed bytes inside one word */
  uint32_t r = 7;
  for (int i = 0; i < 200000; i++) {
    uint64_t a = 0, b = 0;
    for (int k = 0; k < 8; k++) {
      r = r * 1103515245u + 12345u; a |= (uint64_t)((r >> 16) & 0xFF) << (8 * k);
      r = r * 1103515245u + 12345u; b |= (uint64_t)((r >> 16) & 0xFF) << (8 * k);
    }
    uint64_t got = umax8_general(a ^ HI, b ^ HI) ^ HI;
    for (int k = 0; k < 8; k++) {
      int8_t ai = (int8_t)(a >> (8 * k)), bi = (int8_t)(b >> (8 * k));
      if ((int8_t)(got >> (8 * k)) != (ai > bi ? ai : bi)) { bad_g++; break; }
    }
  }
  printf("  exhaustive: cheap %d wrong, general %d wrong\n", bad_c, bad_g);

  pool_scalar(); pool_swar(); pool_general();
  printf("  agree: cheap %s  general %s\n",
         memcmp(out_scalar, out_swar, sizeof out_scalar) ? "NO" : "yes",
         memcmp(out_scalar, out_gen, sizeof out_gen) ? "NO" : "yes");

  enum { REPS = 200 };
  uint64_t t0 = ticks();
  for (int r = 0; r < REPS; r++) pool_scalar();
  double a = (ticks() - t0) * tick_ms() / REPS;
  t0 = ticks();
  for (int r = 0; r < REPS; r++) pool_swar();
  double b = (ticks() - t0) * tick_ms() / REPS;
  t0 = ticks();
  for (int r2 = 0; r2 < REPS; r2++) pool_general();
  double g = (ticks() - t0) * tick_ms() / REPS;
  printf("  scalar %.4f   cheap %.4f (%.2fx)   general %.4f (%.2fx)\n",
         a, b, a / b, g, a / g);
  return 0;
}
