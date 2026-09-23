#!/usr/bin/env bash
# The gate every change to this project goes through.
#
#   regression/gate.sh build   <dir>          compile the current tree into <dir>
#   regression/gate.sh compare <new> <prev>   run both on the board and compare
#
# A build directory holds one object per model *and* the two runtime objects it
# was built with, so a change to the runtime is gated exactly like a change to
# the compiler: `compare` links each side against its own runtime.
#
# `compare` is one board session that does, for every model:
#
#   1. the new object against gemmini.h's own CPU implementation of the same
#      calls, byte for byte, forty runs, one process each;
#   2. the new object against the previous build on the accelerator, byte for
#      byte, forty runs -- so a change that is "equally right" but different is
#      seen, not only one that is wrong;
#   3. both builds timed, alternated A B A B in the same session, minimum of
#      two. Two sessions on this board have read the same object at 150 ms and
#      243 ms; only adjacent readings compare.
#
# Forty because one fault here is intermittent: resadd_i8 at 16x64 writes a row
# from another computation about one inference in four. A single comparison
# passed it for months. A mismatch on a model whose object did not change is
# that fault, not the change -- check before reverting.
#
# Inputs come from regression/models (regression/export/export_all.sh).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
models_dir=${GEMMLIR_MODELS:-$here/models}
build=${GEMMLIR_BUILD:-$repo/build}
cc=${GEMMLIR_RISCV_CC:-riscv64-linux-gnu-gcc}
nfs=${GEMMLIR_NFS:-/srv/nfs/debian-riscv64/tmp}
target=${GEMMLIR_TARGET_DIR:-/mnt2/tmp}
tssh=${GEMMLIR_TSSH:-$HOME/03_gemmini/vivado-risc-v/experiments/tssh.py}
runs=${GEMMLIR_RUNS:-40}
export PATH=$HOME/.local/bin:$PATH
export LLVM_BIN=${LLVM_BIN:-/home/jaemin/05_gemmlir/llvm-project/build/bin}

MODELS=${GEMMLIR_GATE_MODELS:-"efficientnet_b0 regnet_y_400mf shufflenet_v2_x0_5 lstm resnet18 mobilenet_v2 mnasnet0_5 squeezenet1_1 vit_tiny googlenet mobilenet_v3_small resnet50 densenet121 gpt_tiny"}

mains() {  # the dump and time harnesses a model links against
  case "$1" in
    gpt_*) echo "$here/harness/gpt_dump_main.c $here/harness/gpt_time_main.c" ;;
    *)     echo "$here/harness/dump_main.c $here/harness/time_main.c" ;;
  esac
}

cmd_build() {
  local dir=${1:?usage: gate.sh build <dir>}
  mkdir -p "$dir"
  for n in $MODELS; do
    [ -f "$models_dir/${n}_cal.mlir" ] || { echo "$n: no input; run export/export_all.sh"; exit 1; }
    "$repo/scripts/compile.sh" "$models_dir/${n}_cal.mlir" --quantize -o "$dir/$n.o" >/dev/null
    echo "  $n"
  done
  cp "$build/runtime/gemmlir_rt.o" "$build/runtime/gemmlir_rt_cpu.o" "$dir/"
  git -C "$repo" rev-parse --short HEAD > "$dir/SHA"
  git -C "$repo" diff --quiet || echo "+ uncommitted changes" >> "$dir/SHA"
  echo "built into $dir ($(tr '\n' ' ' < "$dir/SHA"))"
}

cmd_compare() {
  local new=${1:?usage: gate.sh compare <new> <prev>} prev=${2:?usage: gate.sh compare <new> <prev>}
  if pgrep -f "python3 .*tssh.py" >/dev/null; then
    echo "another board session is running; one at a time (two took the board off the network)"; exit 1
  fi
  local stage="$nfs/gate"; mkdir -p "$stage"; chmod 777 "$stage"
  for n in $MODELS; do
    read -r dump time <<< "$(mains "$n")"
    link() { "$cc" -O2 -static -s -I"$models_dir" -I"$here/harness" $4 \
               -DDATA_HEADER="\"${n}_data.h\"" -o "$stage/${n}_$1" "$2" $3 -lm; }
    link g   "$dump" "$new/$n.o $new/gemmlir_rt.o"      ""
    link c   "$dump" "$new/$n.o $new/gemmlir_rt_cpu.o"  ""
    link p   "$dump" "$prev/$n.o $prev/gemmlir_rt.o"    ""
    link tn  "$time" "$new/$n.o $new/gemmlir_rt.o"      "-DREPS=3"
    link tp  "$time" "$prev/$n.o $prev/gemmlir_rt.o"    "-DREPS=3"
    chmod 755 "$stage/${n}"_*
  done
  local cmd="mountpoint -q /mnt2 || (echo debian | sudo -S mount --bind / /mnt2) 2>/dev/null
cd $target/gate; : > $target/gate.log
for n in $MODELS; do
  ./\${n}_c /dev/shm/c.bin >/dev/null 2>&1; ./\${n}_p /dev/shm/p.bin >/dev/null 2>&1
  d=0; q=0
  for i in \$(seq 1 $runs); do
    ./\${n}_g /dev/shm/x.bin >/dev/null 2>&1
    cmp -s /dev/shm/x.bin /dev/shm/c.bin || d=\$((d+1))
    cmp -s /dev/shm/x.bin /dev/shm/p.bin || q=\$((q+1))
  done
  echo \"\$n: \$d of $runs vs cpu | \$q of $runs vs previous build\" >> $target/gate.log
done
echo CHECKED >> $target/gate.log
for n in $MODELS; do
  a1=\$(./\${n}_tn 2>&1 | tr '\n' ' '); b1=\$(./\${n}_tp 2>&1 | tr '\n' ' ')
  a2=\$(./\${n}_tn 2>&1 | tr '\n' ' '); b2=\$(./\${n}_tp 2>&1 | tr '\n' ' ')
  echo \"\$n NEW  \$a1 | \$a2\" >> $target/gate.log
  echo \"\$n PREV \$b1 | \$b2\" >> $target/gate.log
done
echo DONE >> $target/gate.log"
  python3 "$tssh" "$cmd" 20000 >/dev/null 2>&1 || true
  python3 "$here/reduce.py" "$nfs/gate.log"
}

case "${1:-}" in
  build)   shift; cmd_build "$@" ;;
  compare) shift; cmd_compare "$@" ;;
  *) sed -n '2,4p' "$0"; exit 2 ;;
esac
