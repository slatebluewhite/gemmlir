// f32 tensor matmul -> forced int8 quantization -> Gemmini call, end to end.
// RUN: gemmlir-opt %s \
// RUN:   --force-quantized-matmul --canonicalize \
// RUN:   --lower-quant-ops --strip-func-quant-types --canonicalize \
// RUN:   --convert-elementwise-to-linalg --canonicalize \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
// RUN:   --buffer-deallocation-pipeline \
// RUN:   --convert-linalg-to-gemmlir --convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata \
// RUN:   --convert-gemmlir-to-llvm --convert-index-to-llvm --convert-arith-to-llvm --convert-math-to-llvm --convert-cf-to-llvm \
// RUN:   --convert-func-to-llvm="use-bare-ptr-memref-call-conv=1" --reconcile-unrealized-casts --canonicalize --cse \
// RUN: | FileCheck %s

// The result tensor becomes a heap buffer returned to the caller.
// CHECK:       llvm.func @matmul_example(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> !llvm.ptr
// Quantization (qcast) is lowered to scalar loops ending in an f32 -> i8 conversion...
// CHECK:         llvm.fptosi
// ...and the int8 matmul itself goes to Gemmini.
// CHECK:         llvm.inline_asm
// CHECK-NEXT:    llvm.call @gemmlir_flush()
// CHECK-NEXT:    llvm.call @tiled_matmul_auto(
// CHECK-NOT:     quant.
// CHECK-NOT:     tensor.
// CHECK-NOT:     linalg.
// CHECK-NOT:     memref.
func.func @matmul_example(%A: tensor<128x128xf32>, %B: tensor<128x256xf32>, %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x256xf32>) outs(%C : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}
