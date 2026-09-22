#!/usr/bin/env bash
# compare.sh <model-dir> [model ...]
#
# One row per model: the same compiled object run twice on the board, once
# linked against the Gemmini runtime and once against `gemmlir_rt_cpu.o`, which
# is the same arithmetic on the host. That is the honest comparison -- not a
# hand-written baseline, the *identical* program with the accelerator taken
# away -- and it is why the outputs can be required to match byte for byte.
#
#   ./examples/showcase/compare.sh <dir>            every <name>_cal.mlir with a driver
#   ./examples/showcase/compare.sh <dir> gmid atr   just those
#
# Needs: GEMMLIR_BUILD, a RISC-V cross compiler, and TSSH pointing at the board.
set -euo pipefail
dir="${1:?usage: compare.sh <model-dir> [model ...]}"; shift || true
: "${GEMMLIR_BUILD:?set GEMMLIR_BUILD}"
: "${TSSH:=/home/jaemin/03_gemmini/vivado-risc-v/experiments/tssh.py}"
: "${NFS:=/srv/nfs/debian-riscv64/tmp}"
: "${TARGET_TMP:=/mnt2/tmp}"
G="$(cd "$(dirname "$0")/../.." && pwd)"
out="$NFS/showcase"; rm -rf "$out"; mkdir -p "$out"; chmod 777 "$out"

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
  for f in "$dir"/*_cal.mlir; do n=$(basename "$f" _cal.mlir); models+=("$n"); done
fi

built=()
for n in "${models[@]}"; do
  [ -f "$dir/${n}_main.c" ] || continue
  "$G/scripts/compile.sh" "$dir/${n}_cal.mlir" --quantize -o "/tmp/showcase_$n.o" >/dev/null
  for v in gemmini cpu; do
    [ "$v" = gemmini ] && rt="$GEMMLIR_BUILD/runtime/gemmlir_rt.o" \
                       || rt="$GEMMLIR_BUILD/runtime/gemmlir_rt_cpu.o"
    riscv64-linux-gnu-gcc -O2 -static -I"$dir" -o "$out/${n}_$v" \
      "$dir/${n}_main.c" "/tmp/showcase_$n.o" "$rt" -lm
  done
  # what the compiler put on the accelerator, for the row
  "$G/scripts/compile.sh" "$dir/${n}_cal.mlir" --quantize --emit=gemmlir 2>/dev/null \
    | grep -o 'gemmlir\.[a-z_0-9]*(' | sed 's/($//;s/(//' | sort | uniq -c \
    | awk '{printf "%s x%s ", $2, $1}' > "/tmp/showcase_$n.calls"
  built+=("$n")
done

printf '%s\n' "${built[@]}" > /tmp/showcase.list
"$TSSH" "cd $TARGET_TMP/showcase && for n in ${built[*]}; do
           echo \"== \$n gemmini\"; ./\${n}_gemmini;
           echo \"== \$n cpu\";     ./\${n}_cpu; done" 6000
