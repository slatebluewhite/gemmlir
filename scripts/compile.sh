#!/usr/bin/env bash
# Lower an MLIR file with linalg.matmul to a RISC-V object that calls the Gemmini runtime.
#
#   ./scripts/compile.sh input.mlir [-o out.o] [--quantize] [--emit=gemmlir|llvm-dialect|llvm-ir|obj]
#
#   default     input is memref-based, i8 x i8 -> i32 (examples/matmul_i8.mlir)
#   --quantize  input is tensor-based f32; force int8 quantization first (examples/matmul_f32_tensor.mlir)
#
# Tools: GEMMLIR_OPT (default $GEMMLIR_BUILD/bin/gemmlir-opt, GEMMLIR_BUILD=./build),
# MLIR_TRANSLATE and LLC (default from $LLVM_BIN, else PATH). MATTR defaults to rv64gc.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
GEMMLIR_BUILD="${GEMMLIR_BUILD:-$HERE/build}"
GEMMLIR_OPT="${GEMMLIR_OPT:-$GEMMLIR_BUILD/bin/gemmlir-opt}"
if [ -n "${LLVM_BIN:-}" ]; then
  MLIR_TRANSLATE="${MLIR_TRANSLATE:-$LLVM_BIN/mlir-translate}"; LLC="${LLC:-$LLVM_BIN/llc}"
else
  MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"; LLC="${LLC:-llc}"
fi
MATTR="${MATTR:-+m,+a,+f,+d,+c}"

emit=obj; out=""; in=""; quantize=0; want_out=0
for a in "$@"; do
  case "$a" in
    --emit=*)   emit="${a#--emit=}" ;;
    --quantize) quantize=1 ;;
    -o)         want_out=1 ;;
    *) if [ "$want_out" = 1 ]; then out="$a"; want_out=0; else in="$a"; fi ;;
  esac
done
[ -n "$in" ] || { sed -n '2,10p' "$0"; exit 2; }
[ -x "$GEMMLIR_OPT" ] || { echo "gemmlir-opt not found at $GEMMLIR_OPT (set GEMMLIR_OPT or GEMMLIR_BUILD)"; exit 1; }

# Pass order follows the dialect hierarchy top-down; see docs/pipeline.md.
FRONT=(--convert-linalg-to-gemmlir)
if [ "$quantize" = 1 ]; then
  FRONT=(--force-quantized-matmul --canonicalize
         --lower-quant-ops --strip-func-quant-types --canonicalize
         --convert-elementwise-to-linalg --canonicalize
         --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map"
         --buffer-deallocation-pipeline
         --convert-linalg-to-gemmlir)
fi
MID=(--convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata)
LOWER=(--convert-gemmlir-to-llvm
       --convert-index-to-llvm --convert-arith-to-llvm --convert-math-to-llvm --convert-cf-to-llvm
       --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1"
       --reconcile-unrealized-casts --canonicalize --cse)

case "$emit" in
  gemmlir)      "$GEMMLIR_OPT" "${FRONT[@]}" "$in" ${out:+-o "$out"} ;;
  llvm-dialect) "$GEMMLIR_OPT" "${FRONT[@]}" "${MID[@]}" "${LOWER[@]}" "$in" ${out:+-o "$out"} ;;
  llvm-ir)      "$GEMMLIR_OPT" "${FRONT[@]}" "${MID[@]}" "${LOWER[@]}" "$in" | "$MLIR_TRANSLATE" --mlir-to-llvmir ${out:+-o "$out"} ;;
  obj)
    "$LLC" --version | grep -q riscv64 || { echo "$LLC has no riscv64 target; rebuild LLVM with RISCV in LLVM_TARGETS_TO_BUILD"; exit 1; }
    out="${out:-${in%.mlir}.o}"
    "$GEMMLIR_OPT" "${FRONT[@]}" "${MID[@]}" "${LOWER[@]}" "$in" | "$MLIR_TRANSLATE" --mlir-to-llvmir \
      | "$LLC" -O2 -march=riscv64 -mattr="$MATTR" -target-abi=lp64d -filetype=obj -o "$out"
    echo "wrote $out  (link with runtime/gemmlir_rt.o)" ;;
  *) echo "unknown --emit=$emit"; exit 2 ;;
esac
