# MLPerf Tiny on Gemmini

The four [MLPerf Tiny v1.0](https://github.com/mlcommons/tiny) benchmarks, taken
from PyTorch through gemmlir to the U280 board. None of them had a Gemmini path
before; none of them needed a line of hand-written kernel code.

| benchmark | model | Gemmini | same object, CPU | speedup | relative L2 | accelerator calls |
|---|---|--:|--:|--:|--:|---|
| image classification | ResNet-8, CIFAR-10 32&times;32 | **5.02 ms** | 9952.03 ms | **1983&times;** | 0.0088 | `conv2d_i8` &times;9 · `resadd_i8` &times;3 · `matmul_i8` &times;2 |
| keyword spotting | DS-CNN, 49&times;10 MFCC | **9.76 ms** | 2486.20 ms | **255&times;** | 0.0135 | `conv2d_i8` &times;4 · `depthwise` &times;4 · `matmul_i8` &times;2 · `matmul_i8_scale` |
| visual wake words | MobileNetV1 0.25&times;, 96&times;96 | **44.22 ms** | 6292.86 ms | **142&times;** | 0.0102 | `conv2d_i8` &times;14 · `depthwise` &times;13 · `matmul_i8` &times;2 |
| anomaly detection | dense autoencoder, 640 | **1.61 ms** | 481.86 ms | **299&times;** | 0.0075 | `matmul_i8` · `matmul_i8_scale` &times;9 |

**Every contraction in all four models is on the accelerator.** Not one
convolution, depthwise convolution, residual add or matrix multiply is left as a
scalar loop.

`relative L2` is against the PyTorch model's own f32 output; the set of models in
`docs/pipeline.md` sits at 0.003-0.015, and these are inside it.

`same object, CPU` is the same compiled object linked against
`runtime/gemmlir_rt_cpu.c` instead of the Gemmini runtime, so the comparison is
the accelerator against the core running the identical program. The two agree
**byte for byte** on every model here, across three inputs replayed three times.

## Running it

```bash
tools/torchenv/bin/python examples/mlperf-tiny/export.py tiny_ic   # or tiny_kws, tiny_vww, tiny_ad
scripts/compile.sh tiny_ic_cal.mlir --quantize -o tiny_ic.o
riscv64-linux-gnu-gcc -O2 -static -DDATA_HEADER='"tiny_ic_data.h"' -DREPS=10 \
    -o tiny_ic_g examples/mlperf-tiny/main.c tiny_ic.o build/runtime/gemmlir_rt.o -lm
```

`sweep.c` is the correctness driver: it writes the whole output for three inputs
replayed three times, so the Gemmini build and the CPU build can be compared
element by element rather than through a norm. See `docs/pipeline.md` for the
board recipe.

## Fidelity to the reference

The architectures follow `mlcommons/tiny`'s reference implementations
(`benchmark/training/*`): `resnet_v1_eembc`, the keyword-spotting DS-CNN,
MobileNetV1 at width 0.25, and the anomaly-detection autoencoder. Parameter
counts come out at 78k / 23k / 214k / 267k against the published 78k / 25k /
221k / 270k -- the differences are the biases dropped in favour of batch norm,
which is what the frontend wants (see `docs/pipeline.md`).

The weights are **random**, not trained: this measures what the compiler and the
hardware do, not what the models predict. Accuracy against a trained checkpoint
is a separate exercise and needs the datasets.
