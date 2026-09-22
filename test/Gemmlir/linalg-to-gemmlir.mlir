// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s
// RUN: not gemmlir-opt --convert-linalg-to-gemmlir %S/Inputs/matmul-i8-out.mlir 2>&1 | FileCheck %s --check-prefix=I8OUT

// i8 x i8 -> i32 accumulation selects matmul_i8.
// CHECK-LABEL: func.func @matmul_i32_out
// CHECK-NOT:     linalg.matmul
// CHECK:         gemmlir.matmul_i8(%arg0, %arg1, %arg2) : (memref<128x128xi8> x memref<128x256xi8>) -> memref<128x256xi32>
func.func @matmul_i32_out(%A: memref<128x128xi8>, %B: memref<128x256xi8>, %C: memref<128x256xi32>) {
  linalg.matmul ins(%A, %B : memref<128x128xi8>, memref<128x256xi8>) outs(%C : memref<128x256xi32>)
  return
}

// C is a plain argument, so its incoming value has to be accumulated into: the
// lowering does that by also passing it as the runtime's bias operand.
// CHECK-NOT:     accumulate = false

// i8 x i8 -> i8 is refused: linalg.matmul wraps, the accelerator's scaled path
// saturates. See Inputs/matmul-i8-out.mlir.
// I8OUT: error: 'linalg.matmul' op i8 output cannot be offloaded
