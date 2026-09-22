/*
 * gemmlir runtime shim.
 *
 * The compiler lowers gemmlir.matmul_i8 to an external call
 *     void tiled_matmul_auto(size_t, size_t, size_t, const elem_t*, const elem_t*,
 *                            const void*, void*, size_t x4, scale_t, scale_t,
 *                            scale_acc_t, int, acc_scale_t, acc_scale_t,
 *                            bool x5, uint8_t, enum tiled_matmul_type_t)
 * which is defined `static` in gemmini.h. Defining EXPOSE_TOP_LEVEL_FNS turns those definitions into ordinary
 * external functions, so this single translation unit provides the symbol.
 *
 * Compile with your RISC-V toolchain and link it together with the object produced
 * by scripts/compile.sh:
 *     riscv64-unknown-linux-gnu-gcc -O2 -march=rv64gc -mabi=lp64d -c gemmlir_rt.c \
 *         -I<gemmini-rocc-tests> -include <your gemmini_params.h>   # -include: optional override
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* gemmini.h's own entry points are renamed out of the way; the ones this file
 * exports wrap them (see "First touch" below). */
#define EXPOSE_TOP_LEVEL_FNS
#define tiled_matmul_auto gemmini_tiled_matmul_auto
#define tiled_conv_auto   gemmini_tiled_conv_auto
#define tiled_conv_stride_auto gemmini_tiled_conv_stride_auto
#define tiled_resadd_auto gemmini_tiled_resadd_auto
#define tiled_conv_dw_auto gemmini_tiled_conv_dw_auto
#define tiled_norm_auto   gemmini_tiled_norm_auto
#include "include/gemmini.h"
#undef tiled_matmul_auto
#undef tiled_conv_auto
#undef tiled_conv_stride_auto
#undef tiled_resadd_auto
#undef tiled_conv_dw_auto
#undef tiled_norm_auto

/* The compiler hard-codes the argument types it emits. Catch a hardware config
 * whose generated gemmini_params.h disagrees, instead of silently passing bits
 * of the wrong type. */
#define GEMMLIR_IS_TYPE(T, want) _Generic((T)0, want: 1, default: 0)
_Static_assert(sizeof(size_t) == 8,
               "gemmlir emits dims/strides as i64; build for a 64-bit RISC-V target");
_Static_assert(GEMMLIR_IS_TYPE(elem_t, int8_t),
               "gemmlir emits int8 inputs; this gemmini_params.h uses a different elem_t");
_Static_assert(GEMMLIR_IS_TYPE(acc_t, int32_t),
               "gemmlir emits int32 accumulators; this gemmini_params.h uses a different acc_t");
_Static_assert(GEMMLIR_IS_TYPE(scale_t, float),
               "gemmlir emits A/B scale factors as f32; this gemmini_params.h uses a non-float scale_t");
_Static_assert(GEMMLIR_IS_TYPE(scale_acc_t, int32_t),
               "gemmlir emits D_scale_factor as i32; this gemmini_params.h uses a different scale_acc_t "
               "(e.g. float when accumulator mvin scaling is enabled). Adjust GemmlirToLLVMPass.cpp or the config.");
_Static_assert(GEMMLIR_IS_TYPE(acc_scale_t, float),
               "gemmlir emits scale/bert_scale as f32; this gemmini_params.h uses a non-float acc_scale_t");

/*
 * First touch: write each output buffer once, the first time it is seen.
 *
 * The accelerator's first write to a page the host has never written does not
 * stick on this board -- the page is still the kernel's shared zero page, and
 * `mlockall` populating it for reading does not change that. Measured on a
 * three-convolution CNN whose buffers are `memref.global`s: the first inference
 * came back at a relative L2 of 0.127 and every one after it at 0.0039, and one
 * `memset` per buffer at first sight makes the first one 0.0039 as well, for no
 * measurable time (8.74 ms against 8.70).
 *
 * Once, not every call: writing the output buffer immediately before every call
 * is a different and worse problem -- Gemmini's writes do not invalidate this
 * board's data cache, so the host reads its own line back. See docs/pipeline.md.
 *
 * And that same non-invalidation applies to the `memset` itself, which leaves
 * the buffer dirty in the data cache. While the accelerator was the only reader
 * of its own output that did not show; as soon as the host reads one -- a
 * residual add split out of a convolution's tail does exactly that -- the first
 * inference reads the zeros back. So the lines are pushed out again afterwards,
 * by walking a scratch region larger than the cache.
 *
 * The scratch region has to be *populated*. Reading untouched `.bss` only ever
 * touches the one shared zero page, which evicts nothing: an earlier attempt at
 * this did that and changed the answer not at all.
 *
 * And the same non-invalidation runs the *other* way: a buffer the host wrote
 * shortly before the accelerator reads it is read past, because the dirty lines
 * are still in the host's L1. So the walk happens on both sides of every call,
 * not only after the first touch.
 *
 * How far to walk: this board's L1 data cache is 16 KB, 64 sets of 4 lines of
 * 64 bytes (the device tree says so), so a contiguous 16 KB of scratch conflicts
 * with every set four times -- once per way. The L2 is a 512 KB inclusive cache
 * and does *not* have to be walked: Gemmini reads through it, and walking it
 * only costs time. Sweeping the walk over twelve inferences of two models that
 * the missing flush breaks, each checked against the same object linked against
 * the CPU runtime:
 *
 *              inverted residual      residual block
 *      0 B     12/12 wrong 1.97 ms    12/12 wrong 1.86 ms
 *      4 KB    12/12 wrong 2.73 ms    11/12 wrong 1.92 ms
 *      8 KB      0/12      2.22 ms    12/12 wrong 2.06 ms
 *     16 KB      0/12      2.59 ms      0/12      2.30 ms
 *     32 KB      0/12      3.80 ms      0/12      3.35 ms
 *    128 KB      0/12      9.96 ms      0/12      8.49 ms
 *
 * The threshold is sharp and it sits exactly at the associativity: two
 * conflicting accesses per set are not enough and four always are, which is
 * what a deterministic replacement policy looks like -- under a random one the
 * failures would thin out gradually instead. So 16 KB, the cache's own
 * capacity, is the smallest walk that displaces every way by construction.
 * Above it nothing improves and everything costs more.
 */
enum { GEMMLIR_L1_BYTES = 16 * 1024 };
enum { GEMMLIR_EVICT_BYTES = 2 * GEMMLIR_L1_BYTES };
static char *gemmlir_evict_pool;
/* The loads have to stay live, and one volatile store at the end is enough to
   keep all eight sums -- and so all 256 loads -- from being folded away. */
static volatile long gemmlir_evict_sink;

static void gemmlir_evict(void) {
  if (!gemmlir_evict_pool) {
    gemmlir_evict_pool = malloc(GEMMLIR_EVICT_BYTES);
    if (!gemmlir_evict_pool)
      return;
    memset(gemmlir_evict_pool, 1, GEMMLIR_EVICT_BYTES);
  }
  /* Eight accumulators, not one.
   *
   * The walk's job is to touch these 256 lines, and it does that whatever it
   * does with what it reads -- but a single `volatile` accumulator chains
   * every load to the one before it, so the walk runs at one cache access per
   * dependency instead of as many as the core will keep in flight. Eight
   * independent sums let them overlap. Measured on the board, same addresses,
   * same order:
   *
   * | | us a walk |
   * |---|---|
   * | one volatile accumulator | 65.13 |
   * | four | 15.63 |
   * | eight | **13.88** |
   *
   * That is 78.7% off an operation PC sampling puts at 5.0% of `vit_tiny` and
   * 2.2% of `densenet121`. Nothing about which lines get displaced changes. */
  long s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  const char *p = gemmlir_evict_pool;
  for (int i = 0; i < GEMMLIR_L1_BYTES; i += 512) {
    s0 += p[i];       s1 += p[i + 64];  s2 += p[i + 128]; s3 += p[i + 192];
    s4 += p[i + 256]; s5 += p[i + 320]; s6 += p[i + 384]; s7 += p[i + 448];
  }
  gemmlir_evict_sink = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
}


/* gemmini.h fences at the end of `tiled_matmul` and `tiled_resadd` and *not* at
 * the end of any convolution, so a convolution's writes are not guaranteed to
 * have landed when it returns. Nothing caught it for a long time because
 * something slow always happened to follow: with a flush on both sides of every
 * call the walk was the delay, and before that every driver replayed one input,
 * so reading the previous call's data gave the same answer. It showed the
 * moment --place-cache-flushes cut the flushes down to one: the inverted
 * residual went wrong on every input, and no walk size fixed it -- only putting
 * something back *between* the calls did, which is what a missing fence looks
 * like rather than a cache that is not empty enough.
 *
 * `fence` is the right instrument: Rocket holds it until the RoCC accelerator
 * reports itself idle. It costs nothing measurable next to the call it follows.
 */
#define gemmlir_fence() asm volatile("fence" ::: "memory")

/* What the generated code calls on either side of an accelerator call, where
 * --place-cache-flushes could not show the flush is redundant. */
void gemmlir_flush(void) { gemmlir_evict(); }

/* Every accelerator output buffer seen so far, so each is prepared once.
 *
 * This was a fixed 256 and a warning past it, which is a cliff:
 * `regnet_y_400mf` makes **634** calls, so 378 of its outputs never got the
 * memset that keeps the accelerator's first write from being lost, and the
 * model came out at 1.33 relative L2 -- wrong, with a line on stderr nobody
 * reads. It grows now, and a failure to grow refuses to go on rather than
 * carrying on quietly. */
/* A hash rather than a list, because this is asked on **every** accelerator
 * call and the answer is almost always "seen it".
 *
 * A linear scan was fine while a model had a few dozen output buffers. Then
 * `--depthwise-as-block-diagonal` split every depthwise layer into a
 * convolution per sixteen channels, each writing its own slice, and
 * EfficientNet went to several hundred distinct pointers. PC sampling put the
 * scan at **6.9% of the model** -- more than the search this file already
 * memoizes. Open addressing, power-of-two table, grown at three quarters.
 *
 * | | ms | |
 * |---|---|---|
 * | `mobilenet_v2` | 79.59 -> **66.43** | -16.5% |
 * | `regnet_y_400mf` | 151.68 -> **133.60** | -11.9% |
 * | `mnasnet0_5` | 54.04 -> **50.01** | -7.5% |
 * | `efficientnet_b0` | 363.45 -> **337.20** | -7.2% |
 * | `resnet18` | 66.93 -> **63.42** | -5.2% |
 * | the set | 2472.85 -> **2405.77** | -2.7% |
 *
 * RegNet, which makes 634 calls and was the model the growing list was written
 * for, gains most after MobileNetV2: it was walking the longest list. All
 * thirteen byte for byte -- the hash answers the same question, only faster. */
static void **gemmlir_touched;
static int gemmlir_touched_count, gemmlir_touched_cap;

static unsigned gemmlir_touched_slot(void **table, int cap, void *p) {
  uintptr_t h = ((uintptr_t)p >> 4) * 0x9E3779B97F4A7C15ull;
  unsigned mask = (unsigned)cap - 1, i = (unsigned)(h >> 40) & mask;
  while (table[i] && table[i] != p)
    i = (i + 1) & mask;
  return i;
}

static void gemmlir_touched_room(void) {
  if (gemmlir_touched_cap && gemmlir_touched_count * 4 < gemmlir_touched_cap * 3)
    return;
  int want = gemmlir_touched_cap ? gemmlir_touched_cap * 2 : 1024;
  void **grown = calloc((size_t)want, sizeof *grown);
  if (!grown) {
    fprintf(stderr, "gemmlir: out of memory tracking accelerator output "
                    "buffers; the first write to a new one would be lost\n");
    abort();
  }
  for (int i = 0; i < gemmlir_touched_cap; i++)
    if (gemmlir_touched[i])
      grown[gemmlir_touched_slot(grown, want, gemmlir_touched[i])] =
          gemmlir_touched[i];
  free(gemmlir_touched);
  gemmlir_touched = grown;
  gemmlir_touched_cap = want;
}

/* True the first time this buffer is seen, and records it. */
static int gemmlir_unseen(void *p) {
  if (!p)
    return 0;
  gemmlir_touched_room();
  unsigned i = gemmlir_touched_slot(gemmlir_touched, gemmlir_touched_cap, p);
  if (gemmlir_touched[i])
    return 0;
  gemmlir_touched[i] = p;
  gemmlir_touched_count++;
  return 1;
}

static void gemmlir_first_touch(void *p, size_t bytes) {
  if (!p || !bytes || !gemmlir_unseen(p))
    return;
  memset(p, 0, bytes);
  gemmlir_evict();
}

/* The symmetric hazard to the first write above, and the one that bites harder.
 *
 * The accelerator is the *first and only* reader of the weights. In a static
 * binary they are file-backed read-only data, faulted in on demand, and nothing
 * on the host side ever looks at them -- `mvin` reads them straight out of the
 * executable's image. Gemmini's TLB miss then asks the core's page-table walker
 * for a page that has no PTE at all, and what comes back is **zeros**: silently,
 * with no fault, no message and no slowdown, so the model's whole answer
 * collapses to zero and nothing says why.
 *
 * Measured on the U280. Every earlier board run went through `sudo chrt`, where
 * `mlockall(MCL_CURRENT)` succeeds and populates every mapping -- that is what
 * hid this for as long as it did. Run the same binary as an ordinary user, where
 * `mlockall` fails on the 8 MB memlock limit, and EfficientNet returns all
 * zeros while the same object against the CPU runtime is exact to L2 0.0200.
 * Touching the read-only segment one byte a page ahead of the call restores it
 * exactly; touching .bss changes nothing, and that is what says the *read* side
 * is the one at fault.
 *
 * One volatile byte a page is all it takes to create the PTE -- bytes/4096 loads
 * against a call that is about to move every one of those bytes through a 16x16
 * array. The last byte goes in by hand because the buffer need not be page
 * aligned, so its final page can start past the last multiple of 4096.
 *
 * Every call, not once per buffer the way `gemmlir_first_touch` is: that table
 * holds 256 pointers and stops protecting anything past it, and the whole point
 * here is that running out of protection is invisible. Measured at 692.03 ms
 * against 688.24 on EfficientNet -- 0.55% to never return zeros.
 */
/* Tried and reverted: touching the whole read-only image once, between `etext`
 * and `__bss_start`, so that no size formula below could be the thing that is
 * short. It does not help -- MobileNetV2 is wrong in 6 of 30 runs with it and 3
 * of 30 without, which is the same number -- and on one binary the range has a
 * hole in it and the walk segfaults. The spans are not what is left.
 */
static void gemmlir_first_read(const void *p, size_t bytes) {
  if (!p || !bytes)
    return;
  const volatile char *q = (const volatile char *)p;
  for (size_t i = 0; i < bytes; i += 4096)
    (void)q[i];
  (void)q[bytes - 1];
}

void tiled_matmul_auto(size_t I, size_t J, size_t K, const elem_t *A,
                       const elem_t *B, const void *D, void *C, size_t sA,
                       size_t sB, size_t sD, size_t sC, scale_t As, scale_t Bs,
                       scale_acc_t Ds, int act, acc_scale_t scale,
                       acc_scale_t bert, bool rep, bool trA, bool trB,
                       bool fullC, bool lowD, uint8_t wA,
                       enum tiled_matmul_type_t t) {
  size_t elems = I ? (I - 1) * sC + J : 0;
  gemmlir_first_touch(C, elems * (fullC ? sizeof(acc_t) : sizeof(elem_t)));
  /* A transposed operand is stored with the other extent down the rows, so the
   * stride belongs to that one; reading the dense formula instead would stop
   * short of the buffer's end and leave its last pages unmapped. */
  gemmlir_first_read(A, (trA ? (K ? (K - 1) * sA + I : 0)
                             : (I ? (I - 1) * sA + K : 0)) * sizeof(elem_t));
  gemmlir_first_read(B, (trB ? (J ? (J - 1) * sB + K : 0)
                             : (K ? (K - 1) * sB + J : 0)) * sizeof(elem_t));
  /* A **repeating** bias is one row that the call broadcasts down the matrix,
   * and one row is all there is in the buffer. Reading `(I-1)*sD + J` of it --
   * the formula for a bias with a row per output row -- reads I times too much,
   * and for a matmul whose bias sits near the end of a mapping that is a
   * segfault rather than a wrong answer. `regnet_y_400mf`, 548 matmuls and a
   * small bias on most of them, crashed on it; every other model's bias happened
   * to have enough behind it. */
  gemmlir_first_read(D, (rep ? J : (I ? (I - 1) * sD + J : 0)) *
                            (lowD ? sizeof(elem_t) : sizeof(acc_t)));
  gemmini_tiled_matmul_auto(I, J, K, A, B, D, C, sA, sB, sD, sC, As, Bs, Ds,
                            act, scale, bert, rep, trA, trB, fullC, lowD, wA, t);
}

void tiled_conv_auto(int batch_size, int in_row_dim, int in_col_dim,
                     int in_channels, int out_channels, int out_row_dim,
                     int out_col_dim, int stride, int input_dilation,
                     int kernel_dilation, int padding, int kernel_dim,
                     bool wrot180, bool trans_output_1203, bool trans_input_3120,
                     bool trans_weight_1203, bool trans_weight_0132,
                     const elem_t *input, const elem_t *weights,
                     const acc_t *bias, elem_t *output, int act,
                     acc_scale_t scale, int pool_size, int pool_stride,
                     int pool_padding, enum tiled_matmul_type_t t) {
  int rows = out_row_dim, cols = out_col_dim;
  if (pool_stride) {
    rows = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    cols = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  }
  gemmlir_first_touch(output, (size_t)batch_size * rows * cols * out_channels *
                                  sizeof(elem_t));
  gemmlir_first_read(input, (size_t)batch_size * in_row_dim * in_col_dim *
                                in_channels * sizeof(elem_t));
  gemmlir_first_read(weights, (size_t)kernel_dim * kernel_dim * in_channels *
                                  out_channels * sizeof(elem_t));
  gemmlir_first_read(bias, (size_t)out_channels * sizeof(acc_t));
  gemmini_tiled_conv_auto(batch_size, in_row_dim, in_col_dim, in_channels,
                          out_channels, out_row_dim, out_col_dim, stride,
                          input_dilation, kernel_dilation, padding, kernel_dim,
                          wrot180, trans_output_1203, trans_input_3120,
                          trans_weight_1203, trans_weight_0132, input, weights,
                          bias, output, act, scale, pool_size, pool_stride,
                          pool_padding, t);
  gemmlir_fence();
}

/* The strided form, which is how a convolution writes into one branch's slice
 * of a concatenation without anyone copying afterwards. The first touch has to
 * follow the same slice: writing `rows * cols * out_stride` bytes would run
 * over the neighbouring branch, which may already hold its result. */
void tiled_conv_stride_auto(int batch_size, int in_row_dim, int in_col_dim,
                            int in_channels, int out_channels, int out_row_dim,
                            int out_col_dim, int stride, int input_dilation,
                            int kernel_dilation, int padding, int kernel_dim,
                            int in_stride, int weight_stride, int out_stride,
                            bool wrot180, bool trans_output_1203,
                            bool trans_input_3120, bool trans_weight_1203,
                            bool trans_weight_0132, const elem_t *input,
                            const elem_t *weights, const acc_t *bias,
                            elem_t *output, int act, acc_scale_t scale,
                            int pool_size, int pool_stride, int pool_padding,
                            enum tiled_matmul_type_t t) {
  int rows = out_row_dim, cols = out_col_dim;
  if (pool_stride) {
    rows = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    cols = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  }
  if (out_stride == out_channels) {
    gemmlir_first_touch(output, (size_t)batch_size * rows * cols * out_channels *
                                    sizeof(elem_t));
  } else if (gemmlir_unseen(output)) {
    size_t pixels = (size_t)batch_size * rows * cols;
    for (size_t p = 0; p < pixels; p++)
      memset(output + p * out_stride, 0, out_channels * sizeof(elem_t));
    gemmlir_evict();
  }
  gemmlir_first_read(input, ((size_t)(batch_size * in_row_dim * in_col_dim - 1) *
                                 in_stride + in_channels) * sizeof(elem_t));
  gemmlir_first_read(weights, ((size_t)(kernel_dim * kernel_dim * in_channels - 1) *
                                   weight_stride + out_channels) * sizeof(elem_t));
  gemmlir_first_read(bias, (size_t)out_channels * sizeof(acc_t));
  gemmini_tiled_conv_stride_auto(
      batch_size, in_row_dim, in_col_dim, in_channels, out_channels,
      out_row_dim, out_col_dim, stride, input_dilation, kernel_dilation,
      padding, kernel_dim, in_stride, weight_stride, out_stride, wrot180,
      trans_output_1203, trans_input_3120, trans_weight_1203, trans_weight_0132,
      input, weights, bias, output, act, scale, pool_size, pool_stride,
      pool_padding, t);
  gemmlir_fence();
}

/* The depthwise call had no wrapper at all, so it got neither the first touch
 * every other accelerator output gets nor the fence above. */
void tiled_conv_dw_auto(int batch_size, int in_row_dim, int in_col_dim,
                        int channels, int out_row_dim, int out_col_dim,
                        int stride, int padding, int kernel_dim, elem_t *input,
                        elem_t *weights, acc_t *bias, elem_t *output, int act,
                        acc_scale_t scale, int pool_size, int pool_stride,
                        int pool_padding, enum tiled_matmul_type_t t) {
  int rows = out_row_dim, cols = out_col_dim;
  if (pool_stride) {
    rows = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    cols = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  }
  gemmlir_first_touch(output,
                      (size_t)batch_size * rows * cols * channels * sizeof(elem_t));
  gemmlir_first_read(input, (size_t)batch_size * in_row_dim * in_col_dim *
                                channels * sizeof(elem_t));
  gemmlir_first_read(weights,
                     (size_t)kernel_dim * kernel_dim * channels * sizeof(elem_t));
  gemmlir_first_read(bias, (size_t)channels * sizeof(acc_t));
  gemmini_tiled_conv_dw_auto(batch_size, in_row_dim, in_col_dim, channels,
                             out_row_dim, out_col_dim, stride, padding,
                             kernel_dim, input, weights, bias, output, act,
                             scale, pool_size, pool_stride, pool_padding, t);
  gemmlir_fence();
}

/* Row-wise normalization of an i32 accumulator.
 *
 * `tiled_norm` configures the softmax constants itself, and layernorm needs
 * none -- it is a mean and a standard deviation. Only those two: `sp_tiled_norm`
 * branches on LAYERNORM and SOFTMAX and has no third case, so an iGELU mvins
 * the accumulator and never mvouts, and the operation verifies against that.
 *
 * The BERT scale the softmax constants come from is hard-coded at 0.05 in
 * `tiled_norm` ("TODO let bert-scale be set by the programmer"), so the
 * approximation does not follow the data's scale -- a property of this call,
 * not a mistake in the caller. */
void tiled_norm_auto(size_t I, size_t J, const acc_t *in, elem_t *out,
                     acc_scale_t C_scale, int act,
                     enum tiled_matmul_type_t t) {
  gemmlir_first_touch(out, I * J * sizeof(elem_t));
  gemmlir_first_read(in, I * J * sizeof(acc_t));
  gemmini_tiled_norm_auto(I, J, in, out, C_scale, act, t);
  gemmlir_fence();
}

void tiled_resadd_auto(size_t I, size_t J, scale_t A_scale, scale_t B_scale,
                       acc_scale_t C_scale, const elem_t *A, const elem_t *B,
                       elem_t *C, bool relu, enum tiled_matmul_type_t t) {
  gemmlir_first_touch(C, I * J * sizeof(elem_t));
  gemmlir_first_read(A, I * J * sizeof(elem_t));
  gemmlir_first_read(B, I * J * sizeof(elem_t));
  gemmini_tiled_resadd_auto(I, J, A_scale, B_scale, C_scale, A, B, C, relu, t);

  /* And again.
   *
   * `resadd_i8` at 16x64 intermittently leaves **exactly one 16-wide row** of
   * its output holding another computation's result -- not zeros, not noise: the
   * same wrong bytes turn up in a different row from one occurrence to the next.
   * About four inferences in twenty-four for a single such call. MobileNetV2 is
   * the only model here that asks for that shape, and it is wrong in **15 runs
   * of 60**; ResNet-18, MNASNet and EfficientNet have no 16x64 and are 0 of 40.
   *
   * The call is not reproducibly wrong: about 2500 invocations outside a model
   * -- fifteen shapes, four alignments, random data, the model's own shapes and
   * scales, inputs written by the accelerator, a matmul issued in front -- did
   * not put a byte out of place. `sp_tiled_resadd` issues one
   * `gemmini_loop_ws` for the whole tile, so there is no software tiling
   * between here and the hardware to blame.
   *
   * Issuing it a second time is **0 of 120**, and costs nothing measurable
   * (MobileNetV2 93.5 -> 93.9 ms): the operation writes its whole output from
   * its whole input, so a second pass is the same answer, and the corruption is
   * evidently drawn afresh -- better than independent draws would give, which
   * is what a fault that only catches a cold configuration looks like.
   *
   * Only where the output does not overlap an input. It never does in anything
   * here -- `--split-residual-add` gives the sum its own buffer -- but an
   * in-place add would read back what the first pass wrote, and then a second
   * pass is not the same answer. */
  {
    size_t n = I * J * sizeof(elem_t);
    const char *c = (const char *)C;
    const char *a = (const char *)A, *b = (const char *)B;
    int overlaps = (c < a + n && a < c + n) || (c < b + n && b < c + n);
    if (!overlaps)
      gemmini_tiled_resadd_auto(I, J, A_scale, B_scale, C_scale, A, B, C, relu, t);
  }
  /* No fence here, and the reason is worth writing down because it was got
   * wrong once: `tiled_resadd_auto` forwards to `tiled_resadd_stride_auto`,
   * which forwards to `tiled_resadd`, and **that** one ends in a
   * `gemmini_fence()`. Reading only the first two and finding no fence is an
   * easy mistake -- one was added here on that reading and taken out again when
   * the third function turned up. The convolutions are the ones that genuinely
   * do not fence. */
}

/*
 * memrefCopy: the one MLIR runtime symbol the generated code can reach for.
 *
 * `memref.copy` lowers to a call to this when it cannot be a plain memcpy, and
 * that happens for ordinary input -- a convolution with padding bufferizes
 * through `tensor.pad`, which copies into the middle of a larger buffer. MLIR
 * ships an implementation in its C runner utils, but that is a host library;
 * this is the same walk, in C, so a RISC-V object links.
 *
 * The descriptors follow MLIR's ABI: an unranked memref is {rank, pointer to a
 * ranked descriptor}, and the ranked one is
 * {allocated, aligned, offset, sizes[rank], strides[rank]}.
 */
#include <stddef.h>

struct gemmlir_unranked_memref {
  int64_t rank;
  void *descriptor;
};

/* What a `linalg.fill` of a uniform byte pattern becomes. A separate symbol
 * rather than libc's `memset` so the two runtimes stay a matched pair and the
 * generated code has one place to look. */
void gemmlir_memset(void *p, int value, size_t bytes) {
  memset(p, value, bytes);
}

/* The run length is not a compile-time constant here, so a plain loop over it
 * is three iterations of loop overhead for three stores. The sizes that matter
 * are few: im2col packs 24 bytes, a channel row 8 or 32. */
static inline void gemmlir_copy_words(uint64_t *to, const uint64_t *from,
                                      size_t words) {
  switch (words) {
  case 1:
    to[0] = from[0];
    return;
  case 2:
    to[0] = from[0];
    to[1] = from[1];
    return;
  case 3:
    to[0] = from[0];
    to[1] = from[1];
    to[2] = from[2];
    return;
  case 4:
    to[0] = from[0];
    to[1] = from[1];
    to[2] = from[2];
    to[3] = from[3];
    return;
  default:
    for (size_t w = 0; w < words; w++)
      to[w] = from[w];
    return;
  }
}

void memrefCopy(int64_t elemSize, struct gemmlir_unranked_memref *srcArg,
                struct gemmlir_unranked_memref *dstArg) {
  int64_t rank = srcArg->rank;

  /* {allocated, aligned, offset, sizes..., strides...} */
  char **srcDesc = (char **)srcArg->descriptor;
  char **dstDesc = (char **)dstArg->descriptor;
  char *srcData = srcDesc[1];
  char *dstData = dstDesc[1];
  const int64_t *srcFields = (const int64_t *)(srcDesc + 2);
  const int64_t *dstFields = (const int64_t *)(dstDesc + 2);
  int64_t srcOffset = srcFields[0], dstOffset = dstFields[0];
  const int64_t *srcSizes = srcFields + 1, *srcStrides = srcSizes + rank;
  const int64_t *dstSizes = dstFields + 1, *dstStrides = dstSizes + rank;

  for (int64_t d = 0; d < rank; d++)
    if (srcSizes[d] == 0)
      return;

  char *srcPtr = srcData + srcOffset * elemSize;
  char *dstPtr = dstData + dstOffset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, (size_t)elemSize);
    return;
  }

  /* Rank is a compile-time-bounded handful in practice; MLIR uses alloca here,
     and a fixed bound keeps this free of malloc on the accelerator path. */
  enum { GEMMLIR_MAX_RANK = 8 };
  if (rank > GEMMLIR_MAX_RANK) {
    printf("memrefCopy: rank %ld exceeds %d\n", (long)rank, GEMMLIR_MAX_RANK);
    exit(1);
  }
  /* The longest suffix of dimensions that is packed in *both* operands is one
     `memcpy`, and the odometer only has to walk what is left. MLIR's own
     runtime copies one element per call whatever the layout is, which is what
     made an ordinary contiguous copy cost as much as it did: a 2048-element i8
     copy is 2048 one-byte `memcpy` calls plus the odometer, against a single
     2048-byte one. A convolution's padding, a concatenation and a residual add
     all bufferize into these. */
  int64_t run = 1, outer = rank;
  while (outer > 0 && srcStrides[outer - 1] == run &&
         dstStrides[outer - 1] == run) {
    run *= srcSizes[outer - 1];
    outer--;
  }
  size_t runBytes = (size_t)(run * elemSize);

  if (outer == 0) {
    memcpy(dstPtr, srcPtr, runBytes);
    return;
  }

  int64_t indices[GEMMLIR_MAX_RANK];
  int64_t srcStep[GEMMLIR_MAX_RANK], dstStep[GEMMLIR_MAX_RANK];
  for (int64_t d = 0; d < outer; d++) {
    indices[d] = 0;
    srcStep[d] = srcStrides[d] * elemSize;
    dstStep[d] = dstStrides[d] * elemSize;
  }

  /* A run of a couple of dozen bytes is not worth a call to `memcpy`: the
   * dispatch costs more than the copy, and im2col's runs are 24 bytes with
   * three thousand of them per pack. Copy those as words instead.
   *
   * Whether every run is 8-byte aligned is decided once, here: the base
   * pointers and every stride have to be, and then no run can drift off. */
  size_t words = runBytes / 8;
  int wordwise = runBytes % 8 == 0 && runBytes <= 64 &&
                 ((uintptr_t)srcPtr % 8) == 0 && ((uintptr_t)dstPtr % 8) == 0;
  for (int64_t d = 0; d < outer && wordwise; d++)
    if ((srcStep[d] % 8) != 0 || (dstStep[d] % 8) != 0)
      wordwise = 0;
  /* A run that is not a whole number of aligned words still beats the call as
   * a byte loop -- im2col over a three-channel image has runs of nine. */
  int bytewise = !wordwise && runBytes <= 64;

  /* The innermost axis is walked here, with its two steps in registers, so the
   * carry loop below runs once per *row* of runs instead of once per run. That
   * and unrolling the word copy are what the run length costing 116 cycles
   * instead of 29 was: `words` is not a constant, so the copy was a loop of
   * three iterations, and the odometer ran for every one of them. */
  int64_t inner = outer - 1;
  int64_t innerCount = srcSizes[inner];
  int64_t innerSrc = srcStep[inner], innerDst = dstStep[inner];

  int64_t readIndex = 0, writeIndex = 0;
  for (;;) {
    const char *from = srcPtr + readIndex;
    char *to = dstPtr + writeIndex;
    if (wordwise) {
      for (int64_t i = 0; i < innerCount; i++) {
        gemmlir_copy_words((uint64_t *)to, (const uint64_t *)from, words);
        from += innerSrc;
        to += innerDst;
      }
    } else if (bytewise) {
      for (int64_t i = 0; i < innerCount; i++) {
        for (size_t b = 0; b < runBytes; b++)
          to[b] = from[b];
        from += innerSrc;
        to += innerDst;
      }
    } else {
      for (int64_t i = 0; i < innerCount; i++) {
        memcpy(to, from, runBytes);
        from += innerSrc;
        to += innerDst;
      }
    }
    if (inner == 0)
      return;
    for (int64_t axis = inner - 1; axis >= 0; axis--) {
      int64_t next = ++indices[axis];
      readIndex += srcStep[axis];
      writeIndex += dstStep[axis];
      if (srcSizes[axis] != next)
        goto next_row;
      if (axis == 0)
        return;
      indices[axis] = 0;
      readIndex -= srcSizes[axis] * srcStep[axis];
      writeIndex -= dstSizes[axis] * dstStep[axis];
    }
  next_row:;
  }
}
