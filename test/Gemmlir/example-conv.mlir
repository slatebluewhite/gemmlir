// The shipped convolution example: ten linalg operations become three calls.
// Kept as a test so the example cannot rot.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir=fuse-pooling=true %S/../../examples/conv_i8.mlir | FileCheck %s

// CHECK-LABEL: func.func @conv_block
// Layer 1 keeps its bias and its relu, and pools on the way out.
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %arg5) bias(%arg2 : memref<16xi32>)
// CHECK-SAME:    act = #gemmlir.act<relu>
// CHECK-SAME:    pool_size = 2 : i64, pool_stride = 2 : i64
// Layer 2 has no bias and no activation.
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg3, %arg6)
// CHECK-NOT:     bias(
// CHECK-SAME:    pool_size = 2 : i64, pool_stride = 2 : i64
// And the residual add is the accelerator's too.
// CHECK:         gemmlir.resadd_i8
// Nothing is left for the CPU.
// CHECK-NOT:     linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:     linalg.pooling_nhwc_max
// CHECK-NOT:     linalg.generic
