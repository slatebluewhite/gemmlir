# Pipeline

![pipeline](img/gemmlir-pipeline.png)

Order: linalg → gemmlir first, then loops/cf, then each dialect → LLVM,
`--convert-func-to-llvm` with bare pointers, `--reconcile-unrealized-casts` last.

## i8 memref input

```bash
gemmlir-opt matmul_i8.mlir \
  --convert-linalg-to-gemmlir \
  --convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata \
  --convert-gemmlir-to-llvm \
  --convert-index-to-llvm --convert-arith-to-llvm --convert-math-to-llvm --convert-cf-to-llvm \
  --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1" \
  --reconcile-unrealized-casts --canonicalize --cse \
| mlir-translate --mlir-to-llvmir \
| llc -O2 -march=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d -filetype=obj -o matmul.o
```

## f32 tensor input (`compile.sh --quantize`)

```bash
gemmlir-opt matmul_f32_tensor.mlir \
  --force-quantized-matmul --canonicalize \
  --lower-quant-ops --strip-func-quant-types --canonicalize \
  --convert-elementwise-to-linalg --canonicalize \
  --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-linalg-to-gemmlir \
  --convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata \
  --convert-gemmlir-to-llvm \
  --convert-index-to-llvm --convert-arith-to-llvm --convert-math-to-llvm --convert-cf-to-llvm \
  --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1" \
  --reconcile-unrealized-casts --canonicalize --cse
```

## CPU reference (no Gemmini)

```bash
mlir-opt matmul_i8.mlir \
  --convert-linalg-to-loops --convert-scf-to-cf --canonicalize --cse \
  --convert-math-to-llvm --convert-arith-to-llvm --expand-strided-metadata --finalize-memref-to-llvm \
  --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1" --convert-cf-to-llvm \
  --reconcile-unrealized-casts
```

## Runtime call

`tiled_matmul_auto(M, N, K, A, B, NULL, C, K, N, N, N, 1.0, 1.0, 1, 0, 1.0, 1.0,
false, transpose_lhs, transpose_rhs, true, false, 0, OS)` — dims/strides `i64`,
pointers `ptr`, `D_scale_factor` `i32`, scales `f32`, flags `i1`. `runtime/gemmlir_rt.c`
static-asserts `gemmini_params.h` against these types. Only `linalg.matmul` on 2-D
static-shape i8 operands is supported; `matmul_i8_scale` (i8 output) has no lowering.
