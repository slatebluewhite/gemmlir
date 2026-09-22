/*
 * gemmlir runtime shim, CPU only.
 *
 * The same surface as gemmlir_rt.c with every matmul forced onto the host, so
 * linking this instead of gemmlir_rt.c measures what the accelerator is buying
 * for *identical* compiled code -- no recompilation, no second quantization, no
 * difference in the surrounding scalar loops.
 *
 * On a U280 Rocket SoC at 62.5 MHz, for a small CNN (two convolutions through
 * im2col, two linear layers, all four matmuls offloaded):
 *
 *     gemmlir_rt.c       35.3 ms/inference
 *     this file          67.3 ms/inference
 *     the same model unquantized, on the CPU    55.1 ms
 *
 * Worth reading twice: the accelerator's own work is a rounding error there.
 * Those matmuls are small enough that tiled_matmul_auto is dominated by its
 * fixed cost -- about 13 us a call -- so well under a millisecond of the 35 is
 * the accelerator. The rest is the code around it: im2col packing, the
 * f32 <-> i8 conversion loops between layers, bias, relu and pooling.
 *
 * Compile it the same way as gemmlir_rt.c.
 */
#define tiled_matmul_auto   gemmini_tiled_matmul_auto
#define tiled_conv_auto     gemmini_tiled_conv_auto
#define tiled_conv_stride_auto gemmini_tiled_conv_stride_auto
#define tiled_conv_dw_auto  gemmini_tiled_conv_dw_auto
#define tiled_norm_auto     gemmini_tiled_norm_auto
#define tiled_resadd_auto   gemmini_tiled_resadd_auto
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "include/gemmini.h"

/* ---- debug probe: a checksum at every call boundary ---------------------
 * Two builds that differ only in a loop shape must produce the same stream of
 * checksums.  The first line that differs says which host loop between two
 * accelerator calls is the one that diverged. */
#include <stdio.h>
static int gp_n = 0;
static FILE *gp_f = NULL;
static void gp_open(void) {
  if (!gp_f) { const char *n = getenv("PROBE"); gp_f = n ? fopen(n, "w") : stderr; }
}
static unsigned long gp_sum(const void *p, long bytes) {
  const unsigned char *b = (const unsigned char *)p;
  unsigned long h = 1469598103934665603UL;
  for (long i = 0; i < bytes; i++) { h ^= b[i]; h *= 1099511628211UL; }
  return h;
}
static void gp(const char *what, const void *p, long bytes) {
  gp_open();
  fprintf(gp_f, "%4d %-40s %8ld %016lx\n", gp_n++, what, bytes, gp_sum(p, bytes));
  fflush(gp_f);
}

#undef tiled_matmul_auto
#undef tiled_conv_auto
#undef tiled_conv_stride_auto
#undef tiled_conv_dw_auto
#undef tiled_norm_auto
#undef tiled_resadd_auto

static void tiled_matmul_body(size_t I, size_t J, size_t K, const elem_t *A,
                       const elem_t *B, const void *D, void *C,
                       size_t sA, size_t sB, size_t sD, size_t sC,
                       scale_t As, scale_t Bs, scale_acc_t Ds, int act,
                       acc_scale_t scale, acc_scale_t bert, bool rep,
                       bool trA, bool trB, bool fullC, bool lowD,
                       uint8_t wA, enum tiled_matmul_type_t t) {
  (void)t;

  /* gemmini.h's CPU path writes elem_t and has no full_C, so that case is done
     here; everything else forwards with the type forced to CPU. */
  if (fullC) {
    const acc_t *bias = (const acc_t *)D;
    acc_t *out = (acc_t *)C;
    for (size_t i = 0; i < I; i++)
      for (size_t j = 0; j < J; j++) {
        acc_t acc = bias ? bias[(rep ? 0 : i) * sD + j] : 0;
        for (size_t k = 0; k < K; k++) {
          elem_t a = trA ? A[k * sA + i] : A[i * sA + k];
          elem_t b = trB ? B[j * sB + k] : B[k * sB + j];
          acc += (acc_t)a * (acc_t)b;
        }
        out[i * sC + j] = acc;
      }
    return;
  }
  gemmini_tiled_matmul_auto(I, J, K, A, B, D, C, sA, sB, sD, sC, As, Bs, Ds,
                            act, scale, bert, rep, trA, trB, fullC, lowD, wA,
                            CPU);
}

/* The convolution and the residual add have no full_C case to stand in for, so
   they only need the type forced. Without them a model whose convolutions reach
   `tiled_conv_auto` cannot be linked against this file at all, and the whole
   point is to compare the *same* object. */
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
  (void)t;
  gemmini_tiled_conv_auto(batch_size, in_row_dim, in_col_dim, in_channels,
                          out_channels, out_row_dim, out_col_dim, stride,
                          input_dilation, kernel_dilation, padding, kernel_dim,
                          wrot180, trans_output_1203, trans_input_3120,
                          trans_weight_1203, trans_weight_0132, input, weights,
                          bias, output, act, scale, pool_size, pool_stride,
                          pool_padding, CPU);
}

/* The strided form, which a convolution writing into one branch's slice of a
   concatenation uses. `conv_cpu` reads the strides the same way the hardware
   does, so this stays an exact reference for that path too. */
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
  (void)t;
  { char lab[96];
    snprintf(lab, sizeof lab, "conv %dx%dx%d k%d s%d p%d ->%dx%dx%d",
             in_row_dim, in_col_dim, in_stride, kernel_dim, stride, padding,
             out_row_dim, out_col_dim, out_stride);
    gp(lab, input, (long)batch_size * in_row_dim * in_col_dim * in_stride); }
  gemmini_tiled_conv_stride_auto(
      batch_size, in_row_dim, in_col_dim, in_channels, out_channels,
      out_row_dim, out_col_dim, stride, input_dilation, kernel_dilation,
      padding, kernel_dim, in_stride, weight_stride, out_stride, wrot180,
      trans_output_1203, trans_input_3120, trans_weight_1203, trans_weight_0132,
      input, weights, bias, output, act, scale, pool_size, pool_stride,
      pool_padding, CPU);
  {
    int prd = pool_stride ? (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1 : out_row_dim;
    int pcd = pool_stride ? (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1 : out_col_dim;
    gp("conv.out", output, (long)batch_size * prd * pcd * out_stride);
  }
}

void tiled_conv_dw_auto(int batch_size, int in_row_dim, int in_col_dim,
                        int channels, int out_row_dim, int out_col_dim,
                        int stride, int padding, int kernel_dim,
                        elem_t *input, elem_t *weights, acc_t *bias,
                        elem_t *output, int act, acc_scale_t scale,
                        int pool_size, int pool_stride, int pool_padding,
                        enum tiled_matmul_type_t t) {
  (void)t;
  gemmini_tiled_conv_dw_auto(batch_size, in_row_dim, in_col_dim, channels,
                             out_row_dim, out_col_dim, stride, padding,
                             kernel_dim, input, weights, bias, output, act,
                             scale, pool_size, pool_stride, pool_padding, CPU);
}

/* `tiled_norm` has no CPU path -- it issues RoCC instructions whatever type it
 * is handed, and the type only decides whether it runs at all. So the reference
 * is written out here instead, transcribed from `matmul_cpu`'s LAYERNORM and
 * SOFTMAX branches in gemmini.h: the same integer formulation, reached the only
 * other way it can be.
 *
 * `bert_scale` is 0.05 because `tiled_norm` hard-codes it ("TODO let bert-scale
 * be set by the programmer"), so the approximation's constants do not follow
 * the data's scale and the reference must not pretend otherwise. */
void tiled_norm_auto(size_t I, size_t J, const acc_t *in, elem_t *out,
                     acc_scale_t C_scale, int act,
                     enum tiled_matmul_type_t t) {
  (void)t;
  acc_t *row = (acc_t *)malloc(J * sizeof(acc_t));
  if (!row)
    return;
  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++)
      row[j] = in[i * J + j];

    if (act == LAYERNORM) {
      acc_t sum = 0;
      for (size_t j = 0; j < J; j++)
        sum += row[j];
      acc_t mean = sum / (acc_t)J;
      acc_t total_err_sq = 0;
      for (size_t j = 0; j < J; j++)
        total_err_sq += (row[j] - mean) * (row[j] - mean);
      acc_t variance = total_err_sq / (acc_t)J;
      acc_t stddev = int_sqrt(variance);
      if (variance == 0)
        stddev = 1;
      for (size_t j = 0; j < J; j++) {
        acc_t centred = row[j] - mean;
        row[j] = ROUND_NEAR_EVEN((double)centred / stddev);
        out[i * J + j] = scale_and_sat(row[j], act, C_scale, 0);
      }
    } else if (act == SOFTMAX) {
      const scale_t a = 0.3585, b = 1.353, c = 0.344;
      const acc_scale_t bert_scale = 0.05;
      const acc_t qln2 = (acc_t)(0.693147 / bert_scale);
      const acc_t qb = b / bert_scale;
      const acc_t qc = c / (a * bert_scale * bert_scale);
      const acc_t qln2_inv = 65536 / qln2;

      acc_t max_q = row[0];
      for (size_t j = 1; j < J; j++)
        if (row[j] > max_q)
          max_q = row[j];

      acc_t sum_exp = 0;
      for (size_t j = 0; j < J; j++) {
        acc_t q = row[j] - max_q;
        acc_t z = (acc_t)(-q * qln2_inv) >> 16;
        if (z > 32)
          z = 32;
        acc_t qp = q + z * qln2;
        acc_t q_exp = (qp + qb) * (qp + qb) + qc;
        row[j] = z >= 31 ? 0 : (q_exp >> z);
        sum_exp += row[j];
      }
      scale_t factor = sum_exp ? (127.f) / (float)sum_exp : 0.f;
      for (size_t j = 0; j < J; j++)
        out[i * J + j] = scale_and_sat(row[j], act, factor, 0);
    } else { /* IGELU is elementwise; the row plays no part. */
      /* `scale_and_sat` is the runtime's own integer GELU, and it wants the
         BERT scale the constants are derived from -- 0.05, the one
         `tiled_norm` hard-codes for softmax, being the only one there is. */
      for (size_t j = 0; j < J; j++)
        out[i * J + j] = scale_and_sat(row[j], act, C_scale, 0.05f);
    }
  }
  free(row);
}

void tiled_resadd_auto(size_t I, size_t J, scale_t A_scale, scale_t B_scale,
                       acc_scale_t C_scale, const elem_t *A, const elem_t *B,
                       elem_t *C, bool relu, enum tiled_matmul_type_t t) {
  (void)t;
  gemmini_tiled_resadd_auto(I, J, A_scale, B_scale, C_scale, A, B, C, relu, CPU);
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

/* The generated code calls this around every accelerator call so the host and
 * Gemmini see each other's writes. Nothing here runs on the accelerator, so
 * there is no cache to displace -- and leaving it out would make the reference
 * build fail to link rather than say so. */
void gemmlir_flush(void) {}

void tiled_matmul_auto(size_t I, size_t J, size_t K, const elem_t *A,
                       const elem_t *B, const void *D, void *C,
                       size_t sA, size_t sB, size_t sD, size_t sC,
                       scale_t As, scale_t Bs, scale_acc_t Ds, int act,
                       acc_scale_t scale, acc_scale_t bert, bool rep,
                       bool trA, bool trB, bool fullC, bool lowD,
                       uint8_t wA, enum tiled_matmul_type_t t) {
  gp("mm.A", A, (long)I * sA);
  tiled_matmul_body(I, J, K, A, B, D, C, sA, sB, sD, sC, As, Bs, Ds, act, scale,
                    bert, rep, trA, trB, fullC, lowD, wA, t);
  gp("mm.out", C, (long)I * sC * (fullC ? (long)sizeof(acc_t) : (long)sizeof(elem_t)));
}

/* The hook `--probe-buffers` calls: a checksum of a whole allocation, tagged. */
void gemmlir_probe(long ptr, long bytes, long id) {
  gp_open();
  fprintf(gp_f, "P%5ld %8ld %016lx\n", id, bytes,
          gp_sum((const void *)(unsigned long)ptr, bytes));
  fflush(gp_f);
}
