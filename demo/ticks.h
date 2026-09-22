/* Timer ticks to milliseconds, without assuming the clock.
 *
 * `read_cycles()` in gemmini_testutils.h is `rdtime`, and rdtime counts the
 * *timebase*, not the CPU. The board went from 62.5 MHz to 50 MHz when it was
 * reprogrammed with the normalization unit, and every driver that multiplied
 * ticks by a hard-coded 1.6 us under-reported by a fifth from that moment on.
 * The device tree knows the number; ask it. */
#ifndef GEMMLIR_TICKS_H
#define GEMMLIR_TICKS_H
#include <stdint.h>
#include <stdio.h>

static inline uint64_t ticks(void) {
  uint64_t t;
  asm volatile("rdtime %0" : "=r"(t));
  return t;
}

static double tick_ms(void) {
  static double ms_per_tick = 0.0;
  if (ms_per_tick == 0.0) {
    unsigned char be[4] = {0, 0, 0, 0};
    FILE *f = fopen("/proc/device-tree/cpus/timebase-frequency", "rb");
    unsigned long hz = 0;
    if (f && fread(be, 1, 4, f) == 4)
      hz = ((unsigned long)be[0] << 24) | ((unsigned long)be[1] << 16) |
           ((unsigned long)be[2] << 8) | be[3];
    if (f)
      fclose(f);
    if (!hz) {
      fprintf(stderr, "timebase-frequency unreadable; timings would be a guess\n");
      hz = 1; /* makes the number obviously wrong rather than quietly wrong */
    }
    ms_per_tick = 1000.0 / (double)hz;
  }
  return ms_per_tick;
}
#endif
