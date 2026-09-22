// The operands carry mvin scales, applied as they are loaded. Unlike `act`,
// which the full_C path silently ignores, these do take effect on both paths --
// checked on hardware against the runtime's MVIN_SCALE.

// RUN: gemmlir-opt --convert-gemmlir-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv --reconcile-unrealized-casts %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s
// RUN: gemmlir-opt --convert-linalg-to-gemmlir %S/Inputs/matmul-plain.mlir \
// RUN: | FileCheck %s --check-prefix=LINALG

// A_scale_factor and B_scale_factor sit right after the four strides.
// CHECK-LABEL: define void @s_a
// CHECK:         call void @tiled_matmul_auto(
// CHECK-SAME:    i64 64, i64 48, i64 48, i64 48, float 5.000000e-01, float 1.000000e+00, i32 1,
func.func @s_a(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi32>
    {accumulate = false, lhs_scale = 5.000000e-01 : f32}
  return
}

// CHECK-LABEL: define void @s_ab
// CHECK:         call void @tiled_matmul_auto(
// CHECK-SAME:    float 5.000000e-01, float 2.500000e-01, i32 1,
func.func @s_ab(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi32>
    {accumulate = false, lhs_scale = 5.000000e-01 : f32, rhs_scale = 2.500000e-01 : f32}
  return
}

// Both mvin scales and the accumulator scale on the quantized path.
// CHECK-LABEL: define void @s_scaled
// CHECK:         call void @tiled_matmul_auto(
// CHECK-SAME:    float 5.000000e-01, float 2.500000e-01, i32 1, i32 0, float 0x3FB99999A0000000,
func.func @s_scaled(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi8>) {
  gemmlir.matmul_i8_scale(%A, %B, %C) : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi8>
    {lhs_scale = 5.000000e-01 : f32, rhs_scale = 2.500000e-01 : f32, scale = 1.000000e-01 : f32}
  return
}

// A plain linalg.matmul is exact integer arithmetic, so it must not pick up a
// requantizing scale: the attributes stay at their 1.0 default and are elided.
// LINALG:     gemmlir.matmul_i8
// LINALG-NOT: lhs_scale
// LINALG-NOT: rhs_scale
