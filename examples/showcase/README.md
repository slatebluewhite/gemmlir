# A PyTorch model Gemmini's own software has no path for

`gemmini-rocc-tests` ships hand-written C for two networks: one call per layer,
weights packed ahead of time. Step outside it and there is no path. **Grouped
convolutions, channel shuffles, dilation, transposed convolutions and attention
do not appear in it at all.**

This directory takes a PyTorch module that uses three of those and runs it on
the U280 board.

## The model

`from_torch.py` builds a small ShuffleNet: a stem, two ShuffleNet-v1 units and a
classifier. Each unit is a 1x1 **grouped** convolution, a **channel shuffle**, a
3x3 **depthwise** convolution, a second grouped convolution and a residual add.

```
tools/torchenv/bin/python examples/showcase/from_torch.py out/
scripts/compile.sh out/shufflenet_cal.mlir --quantize -o shufflenet.o
riscv64-linux-gnu-gcc -O2 -static -Iout -o shufflenet \
    examples/showcase/shufflenet_main.c shufflenet.o build/runtime/gemmlir_rt.o -lm
```

The compiler puts **thirteen** operations on the accelerator -- one
`conv2d_i8`, two `depthwise_conv2d_i8`, nine `matmul_i8` and one
`matmul_i8_scale` -- and on the board:

| | ms/inference | relative L2 against PyTorch |
|---|---|---|
| Gemmini | **14.74** | 0.0023 |
| the same object, CPU runtime | 232.79 | 0.0023 |

## Why the second row is the right comparison

`build/runtime/gemmlir_rt_cpu.o` is the *same arithmetic on the host*: link the
identical compiled object against it and the accelerator is simply taken away.
It is not a hand-written baseline that might be doing something else, which is
why the two can be required to agree **byte for byte** -- `scripts/board-sweep.sh`
checks exactly that, on three inputs replayed three times, for every model.

## Every model, both ways

`compare.sh` builds the table. One row per model, the same object run twice:

```
./examples/showcase/compare.sh <dir-with-the-calibrated-models>
```

| model | Gemmini | CPU runtime | | relative L2 |
|---|---|---|---|---|
| `r20` ResNet-20 | 10.98 ms | 31977 ms | **2912x** | 0.0088 |
| `gr1` grouped residual | 6.35 | 1801 | 284x | 0.0026 |
| `shu` ShuffleNet unit | 2.81 | 744 | 265x | 0.0036 |
| `mbv2` MobileNetV2 | 83.86 | 19975 | 238x | 0.0091 |
| `two` two-headed detector | 2.57 | 611 | 238x | 0.0153 |
| `gap` global-pool head | 1.49 | 345 | 232x | 0.0044 |
| `atr` atrous / ASPP | 9.62 | 1947 | 202x | 0.0152 |
| `res` residual block | 2.03 | 341 | 168x | 0.0097 |
| `up` transposed conv | 1.31 | 177 | 135x | 0.0003 |
| `shf` shuffle block | 2.56 | 275 | 108x | 0.0057 |
| `grp` grouped conv | 6.74 | 690 | 102x | 0.0028 |
| `shub` shuffle + bottleneck | 6.64 | 656 | 99x | 0.0036 |
| `attn` attention block | 1.91 | 75.5 | 40x | 0.0115 |
| `gup` grouped, widening | 10.56 | 333 | 31x | 0.0130 |
| `gmid` bare grouped conv | 11.52 | 224 | 19x | 0.0096 |

26 models, every one offloading, 19x to 2912x, and the relative L2 is the same
on both runtimes in every row. The spread is not the accelerator being
inconsistent: a model that is mostly convolution has most of its work on the
accelerator, and a bare grouped convolution is mostly the int8 conversion at its
own boundary, which runs on the core either way.

## What the frontend needs, which is not obvious

* **PyTorch and torch-mlir live in `tools/torchenv`** (see `docs/pipeline.md`).
* `calibrate.normalize()` on torch-mlir's output. Its bundled MLIR is from early
  2024 and this one is LLVM 22: a `tensor.expand_shape` written then has no
  `output_shape` clause, and a grouped convolution emits three of them per
  layer. Without it the file does not parse.
* `--split-grouped-conv`, then `--conv-nchw-to-nhwc`, then
  `--average-pool-to-contraction` -- **in that order, and before calibrating**.
  Each of them manufactures the operations the quantizer will match, and an
  operation created after the ranges are measured cannot be calibrated: it
  quantizes at the pass's fallback scale, which `--force-quantized-matmul` now
  warns about.
* `bias=False` on the convolutions, with BatchNorm carrying it --
  `--fold-batch-norm` puts it in the weights. torch-mlir also keeps the shapes
  static that way.
