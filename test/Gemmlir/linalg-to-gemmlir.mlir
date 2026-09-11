// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s

// i8 x i8 -> i32 accumulation selects matmul_i8.
// CHECK-LABEL: func.func @matmul_i32_out
// CHECK-NOT:     linalg.matmul
// CHECK:         gemmlir.matmul_i8(%arg0, %arg1, %arg2) : (memref<128x128xi8> x memref<128x256xi8>) -> memref<128x256xi32>
func.func @matmul_i32_out(%A: memref<128x128xi8>, %B: memref<128x256xi8>, %C: memref<128x256xi32>) {
  linalg.matmul ins(%A, %B : memref<128x128xi8>, memref<128x256xi8>) outs(%C : memref<128x256xi32>)
  return
}

// i8 x i8 -> i8 output selects the scaled variant.
// CHECK-LABEL: func.func @matmul_i8_out
// CHECK:         gemmlir.matmul_i8_scale(%arg0, %arg1, %arg2) : (memref<64x64xi8> x memref<64x64xi8>) -> memref<64x64xi8>
func.func @matmul_i8_out(%A: memref<64x64xi8>, %B: memref<64x64xi8>, %C: memref<64x64xi8>) {
  linalg.matmul ins(%A, %B : memref<64x64xi8>, memref<64x64xi8>) outs(%C : memref<64x64xi8>)
  return
}
