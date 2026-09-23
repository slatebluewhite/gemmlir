#!/usr/bin/env bash
# Lower an MLIR file with linalg.matmul to a RISC-V object that calls the Gemmini runtime.
#
#   ./scripts/compile.sh input.mlir [-o out.o] [--quantize] [--from=tosa]
#                        [--dataflow=ws|os] [--emit=gemmlir|llvm-dialect|llvm-ir|obj]
#
#   default      input is memref-based, i8 x i8 -> i32 (examples/matmul_i8.mlir)
#   --quantize   input is tensor-based f32; force int8 quantization first (examples/matmul_f32_tensor.mlir)
#   --from=tosa  input is TOSA; lower it to linalg and bufferize first. What
#                actually reaches gemmlir from a frontend -- see docs/pipeline.md
#                for which TOSA operations currently survive the trip.
#   --dataflow=  Gemmini dataflow to ask the runtime for (default ws). `os` needs a
#                bitstream whose Gemmini was built with Dataflow.OS or Dataflow.BOTH
#                and that computes it correctly -- check before trusting the numbers.
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

emit=obj; out=""; in=""; quantize=0; want_out=0; from=""; dataflow="${DATAFLOW:-ws}"
for a in "$@"; do
  case "$a" in
    --emit=*)     emit="${a#--emit=}" ;;
    --dataflow=*) dataflow="${a#--dataflow=}" ;;
    --from=*)     from="${a#--from=}" ;;
    --quantize) quantize=1 ;;
    -o)         want_out=1 ;;
    *) if [ "$want_out" = 1 ]; then out="$a"; want_out=0; else in="$a"; fi ;;
  esac
done
[ -n "$in" ] || { sed -n '2,10p' "$0"; exit 2; }
[ -x "$GEMMLIR_OPT" ] || { echo "gemmlir-opt not found at $GEMMLIR_OPT (set GEMMLIR_OPT or GEMMLIR_BUILD)"; exit 1; }

case "$from" in
  ""|linalg) ;;
  tosa) ;;
  *) echo "unknown --from=$from (expected tosa)"; exit 2 ;;
esac

# TOSA arrives on tensors, so it needs its own nested pipeline and a
# bufferization before anything below can see a memref.
# --quantize brings its own bufferization (it has to rewrite on tensors first),
# so only bufferize here when it is not in play.
tosa_front() {
  if [ "$from" != tosa ]; then
    cat "$1"
  elif [ "$quantize" = 1 ]; then
    "$GEMMLIR_OPT" --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))" "$1"
  else
    "$GEMMLIR_OPT" --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))" "$1" \
    | "$GEMMLIR_OPT" --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
                     --buffer-deallocation-pipeline
  fi
}

# Pass order follows the dialect hierarchy top-down; see docs/pipeline.md.
CONVERT="--convert-linalg-to-gemmlir=dataflow=$dataflow fuse-pooling=1"
FLUSH=--place-cache-flushes
FRONT=("$CONVERT" --split-matmul-per-requantize
       --unroll-accelerator-loops "$FLUSH")
if [ "$quantize" = 1 ]; then
  FRONT=(--unbatch-single-matmul --canonicalize
         --fold-batch-norm --canonicalize
         # before quantization: the pool becomes a contraction and is then
         # calibrated, quantized and folded like any other layer. Left as a
         # pool it sums in f32, and the convolution feeding it has no i8 result
         # to fold into -- which is why `apb` offloaded one operation of five.
         --raise-spatial-sum-to-pool --canonicalize
         --average-pool-to-contraction --canonicalize
         # a frontend that packs im2col itself leaves the contraction as a
         # `linalg.generic`, which the quantizer does not match. Raising it
         # here is what makes such a model offload at all -- `cnn_i2c`'s first
         # convolution was 16x196x27 multiply-adds in f32.
         --raise-contraction-to-matmul --canonicalize
         --force-quantized-matmul --canonicalize
         # after the operands are quantized and before anything fuses: a layer
         # whose activation the accelerator cannot end in gets a requantization
         # of its own, so the layer itself still offloads.
         --quantize-unfoldable-tails --canonicalize
         --share-branch-quantization --canonicalize
         --lower-quant-ops --round-quantized-casts --strip-func-quant-types --canonicalize
         --convert-elementwise-to-linalg --canonicalize
         # fuse and hoist run to a fixed point by hand: each opens work for the
         # other. Removing the repeats changes the object of every model tried.
         --fuse-elementwise-around-matmul --canonicalize
         --hoist-elementwise-before-gather --canonicalize
         --fuse-elementwise-around-matmul --canonicalize
         --hoist-elementwise-before-gather --canonicalize
         # before --requantize-before-pooling: the requantization it moves lands
         # on whatever the pool reads, and that should be the convolution's
         # output rather than a padded copy of it.
         --drop-unread-padding --canonicalize
         --requantize-before-pooling --canonicalize
         # after --requantize-before-pooling: it lands a quantization on
         # whatever fed the pool, and a padded max-pool's input is a `tensor.pad`
         # -- which is exactly what the hoist above moves across.
         --hoist-elementwise-before-gather --canonicalize
         --fuse-elementwise-around-matmul --canonicalize
         --quantize-bias-into-accumulator --canonicalize
         --pointwise-conv-to-matmul --canonicalize
         --split-residual-add --canonicalize
         # after --split-residual-add: until the residual add is taken apart the
         # convolution feeding it still looks like one with nowhere to
         # requantize, and packing it would cost a `conv2d_i8` and a `resadd_i8`.
         --conv-to-img2col=unfoldable-only=1 --canonicalize
         --materialize-pad-sources
         --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map"
         --buffer-deallocation-pipeline
         "$CONVERT"
         # after the conversion, before the flush placement: sixteen calls
         # become one and the flushes are placed on what is left.
         --depthwise-as-block-diagonal
         # before the flush placement, and after the conversion: the split
         # makes accelerator calls of its own and they have to be placed like
         # any other.
         --split-matmul-per-requantize
         # before the flush placement: a loop it cannot see inside keeps every
         # flush, and a transformer's batch matmul is 24 such loops.
         --unroll-accelerator-loops
         "$FLUSH")
fi
# expand-strided-metadata turns a subview's offset into affine.apply, so affine
# has to be lowered before control flow is flattened.
MID=(# Constants first, so everything after sees folded numbers.
     --saturate-constant-casts --fold-relayout-into-producers
     --table-for-i8-elementwise --hoist-invariant-reciprocal
     # --combine-channel-affine writes (x-m)*r/s as two per-channel coefficients,
     # and its rsqrt(var+eps) is a loop until --fold-constant-elementwise folds
     # it -- so the folder has to come after, here and not in FRONT.
     --combine-channel-affine --fold-scales-into-broadcast --fold-constant-elementwise
     # after the fold: it proves itself exact by evaluating 256 bytes against
     # the coefficients, which have to be constants to be evaluated.
     --batch-norm-in-fixed-point
     --sink-elementwise-into-readers --hoist-broadcast-invariants
     --sink-monotone-below-max-pool
     # Buffers. Static from here on: the accelerator's output addresses must
     # not move between calls on this board.
     --plan-static-buffers
     # before --pack-int8-max-pool, which now reads the strided bands this
     # writes; that is what lets the padded copy go.
     --pool-without-padding
     --gather-to-memref-copy --expand-static-memref-copy
     # --fill-to-memset runs twice. Here it converts the paddings; a max pool's
     # neutral init it refuses, because a later linalg op reads it as an
     # accumulator. After the packer below that pool is an scf.for and the init
     # is no longer a linalg operand, so the second run converts it.
     --fill-only-the-border --fill-to-memset=below-reduction=1
     --order-loops-for-locality
     # before --combine-constant-scales, so a constant divf(c) reaches it as
     # mulf(1/c) and folds with the other scales.
     --reciprocal-for-division --combine-constant-scales --fuse-multiply-add
     --select-to-minmax --drop-clamp-below-relu --clamp-as-range-check
     # Packing: eight channels to a word, and eight table indices to a load.
     --pack-int8-max-pool --pack-int8-table-lookup --relax-float-max-pool
     --fill-to-memset=below-reduction=1
     # Loops. What is left of linalg becomes scf, and the loops are worked on.
     --convert-linalg-to-loops --promote-reduction-accumulator
     --unroll-reduction-windows --unroll-elementwise-loops
     --expand-strided-metadata --lower-affine --convert-scf-to-cf)
# Before func-to-llvm: a returned memref comes back as its *allocated* pointer,
# which is not where the data is when the allocation was aligned.
# The accelerator's buffers must not move between calls on this board, and
# bufferization's allocations are only stable by luck; --plan-static-buffers
# makes them static. That makes the compiled function non-reentrant -- see
# docs/pipeline.md.
LOWER=(--legalize-bare-ptr-returns
       # `math.rsqrt` lowers to a call to `rsqrtf`, which is not a libm function
       # and does not link; a layer norm is full of them. Expanded here, before
       # anything else, because the expansion writes fresh `arith` operations
       # and the conversion for those has to still be ahead of it. **Only**
       # rsqrt: expanding `roundeven` as well produces a `math.copysign` that
       # --convert-math-to-libm marks illegal and cannot lower, and every
       # quantization in every model has a roundeven.
       --math-expand-ops=ops=rsqrt
       --convert-vector-to-llvm
       --convert-gemmlir-to-llvm
       --convert-index-to-llvm --convert-arith-to-llvm
       # libm for what has no LLVM intrinsic: `math.erf` otherwise reaches
       # mlir-translate as an unknown dialect, and a GELU is 0.5x(1 + erf(x/sqrt2))
       # -- ConvNeXt did not compile at all until this was here. After the LLVM
       # conversion, not before: --convert-math-to-libm marks `math.copysign`
       # illegal, and the roundeven expansion above would have produced one.
       --convert-math-to-llvm --convert-math-to-libm --convert-cf-to-llvm
       --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1"
       --reconcile-unrealized-casts --canonicalize --cse
       # last: only mlir-translate reads it, and without it every i64 access is
       # emitted `align 4` and split into two 32-bit halves.
       --set-target-data-layout)

case "$emit" in
  gemmlir)      tosa_front "$in" | "$GEMMLIR_OPT" "${FRONT[@]}" ${out:+-o "$out"} ;;
  llvm-dialect) tosa_front "$in" | "$GEMMLIR_OPT" "${FRONT[@]}" "${MID[@]}" "${LOWER[@]}" ${out:+-o "$out"} ;;
  llvm-ir)      tosa_front "$in" | "$GEMMLIR_OPT" "${FRONT[@]}" "${MID[@]}" "${LOWER[@]}" | "$MLIR_TRANSLATE" --mlir-to-llvmir ${out:+-o "$out"} ;;
  obj)
    "$LLC" --version | grep -q riscv64 || { echo "$LLC has no riscv64 target; rebuild LLVM with RISCV in LLVM_TARGETS_TO_BUILD"; exit 1; }
    out="${out:-${in%.mlir}.o}"
    tosa_front "$in" | "$GEMMLIR_OPT" "${FRONT[@]}" "${MID[@]}" "${LOWER[@]}" | "$MLIR_TRANSLATE" --mlir-to-llvmir \
      | "$LLC" -O2 -march=riscv64 -mattr="$MATTR" -target-abi=lp64d -filetype=obj -o "$out"
    echo "wrote $out  (link with runtime/gemmlir_rt.o)" ;;
  *) echo "unknown --emit=$emit"; exit 2 ;;
esac
