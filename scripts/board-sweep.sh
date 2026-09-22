#!/usr/bin/env bash
# board-sweep.sh <model-dir> [model ...]
#
# Compiles each `<name>_cal.mlir` once and links the *same object* against both
# runtimes -- gemmlir_rt.o (Gemmini) and gemmlir_rt_cpu.o (the same arithmetic
# on the host) -- runs both on the board, and compares the outputs byte for
# byte.
#
# Why byte for byte and not a relative L2: a model can be wrong on the board and
# still land at the same L2 as the reference. `atr` and `atrn` both read 0.0152
# and 0.0150 either way while the accelerator's convolution was corrupting
# elements near the image edge. The L2 check ran for weeks and never saw it.
#
# Why three inputs, each replayed three times: a driver that replays one input
# cannot tell a stale read from a fresh one -- the data is identical either way.
# Three of six models were once wrong on every inference after the first, and
# invisible for exactly that reason.
set -euo pipefail
dir="${1:?usage: board-sweep.sh <model-dir> [model ...]}"; shift || true
: "${GEMMLIR_BUILD:?set GEMMLIR_BUILD}"
: "${TSSH:=/home/jaemin/03_gemmini/vivado-risc-v/experiments/tssh.py}"
: "${NFS:=/srv/nfs/debian-riscv64/tmp}"
: "${TARGET_TMP:=/mnt2/tmp}"
G="$(cd "$(dirname "$0")/.." && pwd)"
out="$NFS/sweep"; mkdir -p "$out"; chmod 777 "$out"   # the target writes here as another user

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
  for f in "$dir"/*_cal.mlir; do n=$(basename "$f" _cal.mlir); models+=("$n"); done
fi

for n in "${models[@]}"; do
  [ -f "$dir/${n}_sweep.c" ] || { echo "skip $n: no $dir/${n}_sweep.c driver"; continue; }
  "$G/scripts/compile.sh" "$dir/${n}_cal.mlir" --quantize -o "/tmp/sweep_$n.o" >/dev/null
  for v in g c; do
    [ "$v" = g ] && rt="$GEMMLIR_BUILD/runtime/gemmlir_rt.o" || rt="$GEMMLIR_BUILD/runtime/gemmlir_rt_cpu.o"
    riscv64-linux-gnu-gcc -O2 -static -I"$dir" -o "$out/${n}_sw$v" "$dir/${n}_sweep.c" "/tmp/sweep_$n.o" "$rt" -lm
  done
done

"$TSSH" "cd $TARGET_TMP/sweep && for g in *_swg; do n=\${g%_swg}; ./\$g $TARGET_TMP/sweep/\$n.g.bin; ./\${n}_swc $TARGET_TMP/sweep/\$n.c.bin; done" 3000

fail=0
for f in "$out"/*.g.bin; do
  n=$(basename "$f" .g.bin)
  if cmp -s "$f" "$out/$n.c.bin"; then
    python3 - "$f" "$n" <<'PY'
import struct, sys
b = open(sys.argv[1], 'rb').read(); n = len(b)//4
v = struct.unpack(f'{n}f', b); NY = n//9
drift = any(v[(r*3+k)*NY:(r*3+k+1)*NY] != v[k*NY:(k+1)*NY] for r in range(3) for k in range(3))
print(f"  {sys.argv[2]:10} {'DRIFTS BETWEEN REPLAYS' if drift else 'matches the CPU reference exactly'}")
PY
  else
    echo "  $n         DIFFERS FROM THE CPU REFERENCE"; fail=1
  fi
done
exit $fail
