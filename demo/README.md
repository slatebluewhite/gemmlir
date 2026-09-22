# The demo

The thirteen-model set this project validates against is exported with **random
weights at 64x64**. That is the right thing for checking a compiler — a wrong
rewrite moves the L2 whatever the weights are — but it cannot classify anything.

This is the other half: torchvision's own **pretrained weights**, **224x224**,
real photographs, and the ImageNet class names carried into the binary so the
board prints what it sees.

    ./run.sh                              # resnet18
    ./run.sh squeezenet1_1 googlenet      # or several

`run.sh` exports (if it has not already), compiles, links and runs on the U280.
**It takes the board for itself** — nothing else may hold a `tssh.py` session
while it runs.

## What it prints

    resnet18 @ 224x224, int8 on Gemmini -- 1000 classes, 4 images

    first inference (weights moved in, tiling searched): 11966 ms

    dog          320.3 ms   relative L2 0.1437   top-1 agrees with torch
        1. Samoyed                   94.9%      torch: Samoyed                   88.5%
        2. Arctic fox                 1.8%      torch: Arctic fox                 4.6%
        3. white wolf                 1.6%      torch: white wolf                 4.4%

Next to every line is what PyTorch's float32 model said, so the cost of the
whole int8 pipeline is visible as a difference in names and percentages rather
than a number nobody can picture.

## Measured, 2026-09-16

| model | top-1 agrees with torch | an image | first inference |
|---|---|---|---|
| squeezenet1_1 | **4/4** | 188 ms | 1.7 s |
| mobilenet_v2 | 3/4 | 233 ms | 5.8 s |
| resnet18 | **4/4** | 320 ms | 12.0 s |
| resnet50 | **4/4** | 710 ms | 27.1 s |
| googlenet | 3/4 | 978 ms | 7.9 s |

**Eighteen of twenty (model, photograph) pairs name the class PyTorch named.**
Both misses are the same photograph — Grace Hopper in uniform, which the float
model itself gives only 20% to 49% — and each has the other's answer second.

The first inference is about forty times the rest: the weights are moved in, the
pages are touched for the first time, and `tiled_conv_*_auto` searches its
tiling before the memo has anything in it. None of that is per image, so the
harness runs one before the clock starts and prints it separately.

## Correctness

The demo's own check is the one printed on every line: **does it name the class
PyTorch named**. The byte-for-byte check against the runtime's CPU reference --
the same object linked against `gemmlir_rt_cpu.o`, forty runs, one process each
-- is the validation harness's gate and runs there on all thirteen models at
64x64 on every change. At 224x224 the scalar reference is about an hour an
image, which is not a demo.

## Files

- `run.sh` — export, compile, run.
- `export_demo.py` — pretrained weights, 224x224, calibrated **on these
  photographs**, so the activation scales are measured on the distribution the
  demo runs. Writes `<model>_demo_cal.mlir` and `<model>_demo_data.h`
  (~11 MB a model, regenerated, not kept).
- `demo_main.c` — top five with class names, PyTorch's top five beside it.
- `qpredict.py` — what the weight quantization alone costs, per tensor against
  per channel, without the board. Per-channel halves the L2 on every model here
  and the pipeline does not do it; that is the next card if accuracy ever
  becomes the problem.
- `img/` — four photographs; add your own and re-export.

## Adding a model

`export_demo.py` needs the model's torchvision weights enum in `WEIGHTS`. A
model whose calibration and IR do not line up one-to-one stops with a count
mismatch rather than guessing — see `scripts/calibrate.py`.
