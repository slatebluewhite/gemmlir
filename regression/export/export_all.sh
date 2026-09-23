#!/usr/bin/env bash
# Regenerate the regression set: <name>_raw.mlir, <name>_cal.mlir and
# <name>_data.h for each model, in the directory given (default ../models).
#
# Random weights on random noise at 64x64, from a fixed seed. That is the right
# input for checking a compiler -- a wrong rewrite moves the relative L2
# whatever the weights are -- and it cannot classify anything; demo/ is the
# half that does.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/../models}
py=${GEMMLIR_TORCH_PYTHON:-/home/jaemin/05_gemmlir/tools/torchenv/bin/python}
mkdir -p "$out"; cd "$out"
for n in resnet18 resnet50 googlenet densenet121 mobilenet_v2 mobilenet_v3_small \
         efficientnet_b0 regnet_y_400mf shufflenet_v2_x0_5 mnasnet0_5 squeezenet1_1; do
  "$py" "$here/tv_models.py" "$n" 64
done
"$py" "$here/vit.py"  vit_tiny
"$py" "$here/lstm.py"
"$py" "$here/gpt.py"  gpt_tiny
