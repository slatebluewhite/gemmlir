# From a PyTorch module to RISC-V, one stage at a time

`python3 walkthrough.py` writes every file here. It reads the pass lists out of
`scripts/compile.sh` rather than restating them, so this cannot drift from the
pipeline it documents.

The subject is small on purpose — two convolutions, a batch norm on each, a
residual add and a max-pool — but everything that gives this compiler trouble is
in it: a norm that has to be folded away, an activation that has to be fused, a
join the accelerator cannot write into, and a pool.

```python
def forward(self, x):
    h = torch.relu(self.bn1(self.conv1(x)))
    return self.pool(torch.relu(h + self.bn2(self.conv2(h))))
```

| | file | lines | what happened |
|---|---|---|---|
| 1 | `01_torch_mlir.mlir` | 81 | torch-mlir: the module becomes loops over tensors |
| 2 | `02_nhwc.mlir` | 90 | channels move last |
| 3 | `03_calibrated.mlir` | 90 | each contraction is told the range of its input |
| 4 | `04_quantized.mlir` | 96 | f32 becomes i8; the norms are gone |
| 5 | `05_memref.mlir` | 117 | tensors become buffers |
| 6 | `06_accelerator.mlir` | 91 | **two calls to the accelerator** |
| 7 | `07_host_loops.mlir` | 903 | what is left becomes scalar loops, optimized |
| 8 | `08_llvm_dialect.mlir` | 1094 | one dialect, no structure left |
| 9 | `09_llvm.ll` | 1195 | LLVM IR |
| 10 | `10_riscv.s` | 1630 | RISC-V |

## 1 — torch-mlir

A convolution is a named operation and the batch norm is not: it arrives as
arithmetic, four operations in a loop body.

```mlir
linalg.conv_2d_nchw_fchw ins(%padded, %cst_6 : tensor<1x8x18x18xf32>, ...)
...
  %17 = arith.subf %in, %in_13 : f32      // x - mean
  %18 = arith.mulf %17, %16   : f32       // * rsqrt(var + eps)
  %19 = arith.mulf %18, %in_11 : f32      // * gamma
  %20 = arith.addf %19, %in_12 : f32      // + beta
```

Nothing here knows about an accelerator yet.

## 2 — channels last

`conv_2d_nchw_fchw` becomes `conv_2d_nhwc_hwcf`. Gemmini reads a pixel's
channels consecutively; no frontend emits that layout, so it is made here.

## 3 — calibration

The model is run and the range of each contraction's input is measured, then
written onto it:

```mlir
gemmlir.activation_scale = 2.866964265e-02 : f64
```

A weight's scale can be read off the constant, but an activation's cannot — it
depends on the data. Getting this wrong is not a small error: a fixed scale put
a PyTorch MLP at 0.49 relative L2 where a measured one puts it at 0.012.

## 4 — into int8

Two things happen that are worth separating.

**The batch norm disappears.** `(x - mean) * rsqrt(var + eps) * gamma + beta`
applied to a convolution's output is an affine function of each output channel,
so it folds into that convolution's weights and bias. The four arithmetic
operations above are simply not in this file. A norm left standing would also
block the requantization fusion behind it, so this is worth more than the
arithmetic it removes.

**The constants become i8 at compile time.** A weight's quantization does not
depend on the input, so it is done once here rather than on every inference:

```
tensor<144x16xi8>        the second convolution, already im2col'd and quantized
tensor<1x16x16x8xi8>     activations
```

## 5 — buffers

`tensor` becomes `memref`: values become addresses. Nothing moves onto the
accelerator yet, but now there is something with an address to hand it.

## 6 — the accelerator

```mlir
gemmlir.conv2d_i8(%alloc_4, %2, %alloc_7) bias(%1 : memref<16xi32>)
    {act = #gemmlir.act<relu>, padding = 1 : i64, scale = 0.00155569159 : f32}

gemmlir.matmul_i8(%collapse_shape, %0, %collapse_shape_11)
    {accumulate = false}
```

Read the first one carefully: the convolution, **its bias, its requantization
scale and its relu are one operation**. In stage 1 those were a conv, a generic
adding the bias, a generic doing the norm and a generic doing the relu — four
passes over the data, three of them writing an intermediate buffer. Here they
are one call, and the intermediates never exist.

The second convolution went through im2col and became a matmul, because its
shape made that cheaper.

What stays behind is the max-pool, a transpose, and three small elementwise
loops — the accelerator has no instruction for any of them.

## 7 — the host loops

The biggest file, and where most of this project's work went. Everything the
accelerator cannot do becomes scalar loops, and then about forty passes work on
those loops: loop order chosen for locality, leaves unrolled eight ways,
reduction windows straightened, clamps rewritten as one unsigned comparison,
fills turned into `memset`, buffers made static because the accelerator's
output addresses must not move between calls.

On the real model set this stage is where the time is — under 1 ms of a 35 ms
inference was ever the accelerator.

## 8–10 — out of MLIR

One dialect, then LLVM IR, then RISC-V. The calls that survive to the assembly
are the whole story:

```
call tiled_conv_stride_auto     the first convolution
call tiled_matmul_auto          the second
call gemmlir_flush     x2       the cache, before the accelerator reads
call gemmlir_memset    x2       padding
```

Four calls into the runtime, from fourteen `linalg` operations.

## Regenerating

```bash
cd docs/reference/lowering && python3 walkthrough.py
```

Needs the torch environment (torch-mlir, torchvision) and a built
`build/bin/gemmlir-opt`.
