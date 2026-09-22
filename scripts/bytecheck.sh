#!/usr/bin/env bash
# Byte-identity against the runtime's own CPU implementation, forty runs.
#
# The same compiled object is linked twice -- once against runtime/gemmlir_rt.o
# and once against runtime/gemmlir_rt_cpu.o -- and the two outputs are compared
# byte for byte. That is the project's strongest check: the CPU arm is not a
# separate implementation that might be computing something else, it is the same
# object with the accelerator taken away.
#
# **Forty runs, not one.** `resadd_i8` at 16x64 leaves one row of its output
# holding another computation's result in about one inference in four, and a
# single comparison passed it for months. Anything that talks to the accelerator
# can fail this way; a pass has to be a pass over a sample.
#
# Usage: bytecheck.sh <object.o> <data-header.h> [runs]
#   The harness expects `float *forward(float *)` and a header defining
#   `torch_x` and `torch_y`; see scripts/bytecheck_main.c.
set -euo pipefail

obj=${1:?usage: bytecheck.sh <object.o> <data-header.h> [runs]}
header=${2:?usage: bytecheck.sh <object.o> <data-header.h> [runs]}
runs=${3:-40}

here=$(cd "$(dirname "$0")" && pwd)
build=${GEMMLIR_BUILD:-$here/../build}
cc=${GEMMLIR_RISCV_CC:-riscv64-linux-gnu-gcc}
nfs=${GEMMLIR_NFS:-/srv/nfs/debian-riscv64/tmp}
target=${GEMMLIR_TARGET_DIR:-/mnt2/tmp}
tssh=${GEMMLIR_TSSH:-$HOME/03_gemmini/vivado-risc-v/experiments/tssh.py}

name=bc_$(basename "${obj%.o}")
for arm in g c; do
  [ "$arm" = g ] && rt=$build/runtime/gemmlir_rt.o || rt=$build/runtime/gemmlir_rt_cpu.o
  "$cc" -O2 -static -s -I"$(dirname "$header")" -DDATA_HEADER="\"$(basename "$header")\"" \
        -o "$nfs/${name}_$arm" "$here/bytecheck_main.c" "$obj" "$rt" -lm
  chmod 755 "$nfs/${name}_$arm"
done

python3 "$tssh" "cd $target && ./${name}_c /dev/shm/ref.bin >/dev/null &&
  d=0; for i in \$(seq 1 $runs); do ./${name}_g /dev/shm/got.bin >/dev/null;
  cmp -s /dev/shm/got.bin /dev/shm/ref.bin || d=\$((d+1)); done;
  echo \"$name: \$d of $runs differ\"" 5400 | grep 'differ'
