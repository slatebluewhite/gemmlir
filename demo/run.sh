#!/usr/bin/env bash
# One command: export a pretrained torchvision classifier at 224x224, calibrate
# it on the photographs in img/, compile it, and run it on the board.
#
#   ./run.sh                      resnet18
#   ./run.sh squeezenet1_1 resnet18 googlenet
#
# Everything the export writes (~11 MB a model) is regenerated, so it is not
# kept; a model already exported is not exported again.
set -eu
export PATH=$HOME/.local/bin:$PATH LLVM_BIN=/home/jaemin/05_gemmlir/llvm-project/build/bin
HERE=$(cd "$(dirname "$0")" && pwd)
G=$(cd "$HERE/.." && pwd)
RT=$G/build/runtime
PY=/home/jaemin/05_gemmlir/tools/torchenv/bin/python
TSSH=/home/jaemin/03_gemmini/vivado-risc-v/experiments/tssh.py
BOARD=/srv/nfs/debian-riscv64/tmp
MODELS="${*:-resnet18}"

for n in $MODELS; do
  if [ ! -f "$HERE/${n}_demo_cal.mlir" ]; then
    echo "== exporting $n with its pretrained weights"
    (cd "$HERE" && timeout 3000 "$PY" export_demo.py "$n" 224)
  fi
  echo "== compiling $n"
  timeout 4000 "$G/scripts/compile.sh" "$HERE/${n}_demo_cal.mlir" --quantize -o "/tmp/demo_$n.o" >/dev/null
  riscv64-linux-gnu-gcc -O2 -static -s -I"$HERE" -I"$HERE" \
    -DDATA_HEADER="\"${n}_demo_data.h\"" -DMODEL_NAME="\"$n @ 224x224\"" \
    -o "$BOARD/demo_${n}_g" "$HERE/demo_main.c" "/tmp/demo_$n.o" "$RT/gemmlir_rt.o" -lm
  chmod 755 "$BOARD/demo_${n}_g"
done

echo "== running on the board (one session; nothing else may hold it)"
cmd="cd /mnt2/tmp"
for n in $MODELS; do cmd="$cmd; ./demo_${n}_g"; done
python3 "$TSSH" "$cmd" 9000 2>&1 | grep -v "^Connection"
