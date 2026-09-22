#!/usr/bin/env python3
"""A PyTorch unit Gemmini's own software has no path for, taken to the board.

`gemmini-rocc-tests` ships hand-written C for two networks, one call per layer,
with the weights packed ahead of time. A ShuffleNet block is outside it in three
separate ways: the 1x1 convolutions are **grouped**, the 3x3 is **depthwise**,
and between them sits a **channel shuffle**, which is a permutation no kernel in
that library expresses.

This writes the whole thing out -- the linalg the frontend produces, the scales
measured by running the model, and a C header with the reference output -- and
prints the two commands that turn it into a board binary.

    tools/torchenv/bin/python examples/showcase/from_torch.py [outdir]
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import torch
import torch.nn as nn
import torch_mlir
from calibrate import activation_ranges, annotate, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OPT = os.path.join(ROOT, "build", "bin", "gemmlir-opt")


class ShuffleUnit(nn.Module):
    """A ShuffleNet-v1 unit, which `gemmini-rocc-tests` has no path for.

    1x1 **grouped** convolution, a **channel shuffle**, a 3x3 **depthwise**
    convolution, another 1x1 grouped convolution, and a residual add. The
    accelerator's own library ships hand-written C for two networks, one call
    per layer, with the weights packed ahead of time -- and none of the four
    things in that sentence appear in it.

    No bias anywhere: BatchNorm carries it in this shape of network and
    `--fold-batch-norm` puts it in the weights before anything is quantized.
    """

    def __init__(self, channels=16, groups=2, bottleneck=8):
        super().__init__()
        self.groups = groups
        self.expand = nn.Conv2d(channels, bottleneck, 1, groups=groups, bias=False)
        self.en = nn.BatchNorm2d(bottleneck)
        self.spatial = nn.Conv2d(bottleneck, bottleneck, 3, padding=1,
                                 groups=bottleneck, bias=False)
        self.sn = nn.BatchNorm2d(bottleneck)
        self.project = nn.Conv2d(bottleneck, channels, 1, groups=groups, bias=False)
        self.pn = nn.BatchNorm2d(channels)

    def shuffle(self, x):
        """Interleave the channels the groups kept apart, so the next grouped
        convolution sees all of them."""
        n, c, h, w = 1, x.shape[1], x.shape[2], x.shape[3]
        return (x.view(n, self.groups, c // self.groups, h, w)
                 .transpose(1, 2)
                 .reshape(n, c, h, w))

    def forward(self, x):
        y = torch.relu(self.en(self.expand(x)))
        y = self.shuffle(y)
        y = self.sn(self.spatial(y))
        y = self.pn(self.project(y))
        return torch.relu(x + y)


class ShuffleNet(nn.Module):
    """A small network around the unit: a stem, two stages, a global average
    pool and a classifier -- the shape a real model has, so the cost of
    converting at the f32 boundary is amortised over some work rather than
    being all of it.
    """

    def __init__(self, channels=16, classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.sn = nn.BatchNorm2d(channels)
        self.stage1 = ShuffleUnit(channels)
        self.stage2 = ShuffleUnit(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, classes)

    def forward(self, x):
        x = torch.relu(self.sn(self.stem(x)))
        x = self.stage2(self.stage1(x))
        return self.fc(self.pool(x).flatten(1))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    os.makedirs(out, exist_ok=True)
    name = "shufflenet"

    torch.manual_seed(7)
    model = ShuffleNet().eval()
    example = torch.randn(1, 3, 16, 16)

    # The ranges have to be measured on the model as written: a weight's scale
    # is in the constant, an activation's is only known by running it.
    ranges = activation_ranges(model, [example])
    with torch.no_grad():
        reference = model(example)

    # `normalize` is not cosmetic: torch-mlir's bundled MLIR is from early
    # 2024 and this one is LLVM 22, and a `tensor.expand_shape` written then
    # has no `output_shape` clause -- which is exactly what a **grouped**
    # convolution emits, three of them per layer. Without this the file does
    # not parse at all.
    raw = os.path.join(out, name + "_raw.mlir")
    with open(raw, "w") as f:
        f.write(normalize(str(torch_mlir.compile(model, example,
                                                 output_type="linalg-on-tensors"))))

    # NHWC before calibrating, because that is the layout the accelerator's
    # convolutions read and the annotation has to land on the operations the
    # quantizer will actually see.
    # The global average pool has to become a contraction here too, before the
    # scales are written -- and *after* the layout change, because the pass
    # matches the NHWC pooling and the model arrives NCHW.
    # `--average-pool-to-contraction` makes it
    # `ones(1, P) x image(P, C)`, and only then is there an operation for
    # `calibrate.py` to annotate. Left until `compile.sh`, it is created after
    # calibration and quantizes at the pass's fallback scale.
    #
    # A grouped convolution arrives as the 5-D `linalg.conv_2d_ngchw_gfchw`,
    # which no accelerator op matches; `--split-grouped-conv` makes it G
    # ordinary convolutions over channel slices. Then NHWC, because that is
    # the layout `tiled_conv_auto` reads -- and both have to happen *before*
    # the scales are written, or the annotation lands on operations the
    # quantizer never sees.
    laid_out = subprocess.run(
        [OPT, "--split-grouped-conv", "--canonicalize",
              "--conv-nchw-to-nhwc", "--canonicalize",
              "--average-pool-to-contraction", "--canonicalize", raw],
        capture_output=True, text=True)
    if laid_out.returncode:
        sys.stderr.write(laid_out.stderr[:4000])
        return 1
    with open(os.path.join(out, name + "_cal.mlir"), "w") as f:
        f.write(annotate(laid_out.stdout, ranges))

    with open(os.path.join(out, name + "_data.h"), "w") as f:
        for label, tensor in (("x", example), ("y", reference)):
            f.write("static const float torch_%s[%d] = {%s};\n"
                    % (label, tensor.numel(),
                       ",".join("%.9ef" % v for v in tensor.flatten())))

    print("%d layers calibrated; wrote %s_{raw,cal}.mlir and %s_data.h into %s"
          % (len(ranges), name, name, out))
    print()
    print("  # to the board:")
    print("  scripts/compile.sh %s/%s_cal.mlir --quantize -o %s.o" % (out, name, name))
    print("  riscv64-linux-gnu-gcc -O2 -static -I%s -o %s \\" % (out, name))
    print("      examples/showcase/shufflenet_main.c  # (%s) %s.o build/runtime/gemmlir_rt.o -lm"
          % (name, name))
    print()
    print("  # and the same object with the accelerator taken away, which is")
    print("  # what the speedup is measured against:")
    print("  ... build/runtime/gemmlir_rt_cpu.o ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
