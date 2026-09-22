// The runtime's D operand can carry a bias instead of the accumulated output.
// Its shape says how it is read: (M, N) elementwise, (1, N) broadcast down the
// matrix, which is the runtime's repeating_bias.

// RUN: gemmlir-opt --convert-gemmlir-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv --reconcile-unrealized-casts %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s
// RUN: gemmlir-opt %s | gemmlir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: not gemmlir-opt %S/Inputs/matmul-bias-and-accumulate.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=BOTH
// RUN: not gemmlir-opt %S/Inputs/matmul-bias-bad-shape.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=SHAPE

// D is the bias, stride_D is its own row stride, repeating_bias is false.
//                                                 A       B       D       C
// CHECK-LABEL: define void @b_full
// CHECK:         call void @tiled_matmul_auto(i64 32, i64 48, i64 64, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[D:[0-9]+]], ptr %{{[0-9]+}}, i64 64, i64 48, i64 48, i64 48,
// CHECK-SAME:    i1 false, i1 false, i1 false, i1 true, i1 false,
// ROUNDTRIP: gemmlir.matmul_i8(%arg0, %arg1, %arg3) bias(%arg2 : memref<32x48xi32>)
func.func @b_full(%A: memref<32x64xi8>, %B: memref<64x48xi8>,
                  %D: memref<32x48xi32>, %C: memref<32x48xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) bias(%D : memref<32x48xi32>)
    : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi32> {accumulate = false}
  return
}

// A single row sets repeating_bias.
// CHECK-LABEL: define void @b_row
// CHECK:         call void @tiled_matmul_auto(
// CHECK-SAME:    i1 true, i1 false, i1 false, i1 true, i1 false,
func.func @b_row(%A: memref<32x64xi8>, %B: memref<64x48xi8>,
                 %D: memref<1x48xi32>, %C: memref<32x48xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) bias(%D : memref<1x48xi32>)
    : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi32> {accumulate = false}
  return
}

// The quantized shape: i32 bias into a scaled i8 result, so full_C is false.
// CHECK-LABEL: define void @b_scaled
// CHECK:         call void @tiled_matmul_auto(
// CHECK-SAME:    i32 0, float 0x3FA99999A0000000, float 1.000000e+00, i1 true, i1 false, i1 false, i1 false, i1 false,
func.func @b_scaled(%A: memref<32x64xi8>, %B: memref<64x48xi8>,
                    %D: memref<1x48xi32>, %C: memref<32x48xi8>) {
  gemmlir.matmul_i8_scale(%A, %B, %C) bias(%D : memref<1x48xi32>)
    : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi8> {scale = 5.000000e-02 : f32}
  return
}

// BOTH: error: 'gemmlir.matmul_i8' op bias and accumulate cannot both be set
// SHAPE: error: 'gemmlir.matmul_i8' op bias must be 32x48 or 1x48, got 'memref<8x48xi32>'
