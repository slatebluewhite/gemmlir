// Convolution lowers to tiled_conv_stride_auto / tiled_conv_dw_auto. The strided
// form is the one always used: it is `tiled_conv_auto` plus the distance between
// two pixels in each of the input and the output, which for an ordinary buffer
// is just the channel count and for one branch's slice of a concatenation is the
// joined width. The runtime takes
// plain ints, so the output extents and the kernel size are computed here.

// RUN: gemmlir-opt --convert-gemmlir-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv --reconcile-unrealized-casts %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s
// RUN: gemmlir-opt %s | gemmlir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: not gemmlir-opt %S/Inputs/conv-bad-shape.mlir 2>&1 | FileCheck %s --check-prefix=BAD

// Both runtime symbols are declared once, at module scope.
// CHECK-DAG: declare void @tiled_conv_stride_auto(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i1, i1, i1, ptr, ptr, ptr, ptr, i32, float, i32, i32, i32, i32)
// CHECK-DAG: declare void @tiled_conv_dw_auto(i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, i32, float, i32, i32, i32, i32)

// 14x14 with a 3x3 kernel and padding 1 keeps its extent; bias is passed and the
// relu becomes act = 1.
//                       batch  H       W       C       F       OH      OW      stride  in_dil  k_dil   pad     K
// CHECK-LABEL: define void @conv_bias_relu
// CHECK:         call void @tiled_conv_stride_auto(i32 1, i32 14, i32 14, i32 16, i32 32, i32 14, i32 14, i32 1, i32 1, i32 1, i32 1, i32 3, i32 16, i32 32, i32 32,
// CHECK-SAME:    i1 false, i1 false, i1 false, i1 false, i1 false,
// CHECK-SAME:    i32 1, float 0x3F999999A0000000, i32 0, i32 0, i32 0, i32 1)
// ROUNDTRIP: gemmlir.conv2d_i8(%arg0, %arg1, %arg3) bias(%arg2 : memref<32xi32>)
func.func @conv_bias_relu(%in: memref<1x14x14x16xi8>, %f: memref<3x3x16x32xi8>,
                          %b: memref<32xi32>, %out: memref<1x14x14x32xi8>) {
  gemmlir.conv2d_i8(%in, %f, %out) bias(%b : memref<32xi32>)
      {padding = 1 : i64, scale = 2.500000e-02 : f32, act = #gemmlir.act<relu>}
      : (memref<1x14x14x16xi8>, memref<3x3x16x32xi8>, memref<1x14x14x32xi8>)
  return
}

// Stride 2 with no padding: (16 - 3)/2 + 1 = 7. No bias means a null pointer.
// CHECK-LABEL: define void @conv_stride2
// CHECK:         call void @tiled_conv_stride_auto(i32 1, i32 16, i32 16, i32 16, i32 16, i32 7, i32 7, i32 2, i32 1, i32 1, i32 0, i32 3, i32 16, i32 16, i32 16,
// CHECK-SAME:    ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr null, ptr %{{[0-9]+}},
func.func @conv_stride2(%in: memref<1x16x16x16xi8>, %f: memref<3x3x16x16xi8>,
                        %out: memref<1x7x7x16xi8>) {
  gemmlir.conv2d_i8(%in, %f, %out) {stride = 2 : i64, scale = 5.000000e-02 : f32}
      : (memref<1x16x16x16xi8>, memref<3x3x16x16xi8>, memref<1x7x7x16xi8>)
  return
}

// Pooling is fused: the runtime still gets the convolution's own 14x14 extent,
// and the pool parameters shrink it to the 7x7 the result memref holds.
// CHECK-LABEL: define void @conv_pool
// CHECK:         call void @tiled_conv_stride_auto(i32 1, i32 14, i32 14, i32 16, i32 16, i32 14, i32 14,
// CHECK-SAME:    i32 2, i32 2, i32 0, i32 1)
func.func @conv_pool(%in: memref<1x14x14x16xi8>, %f: memref<3x3x16x16xi8>,
                     %out: memref<1x7x7x16xi8>) {
  gemmlir.conv2d_i8(%in, %f, %out)
      {padding = 1 : i64, scale = 2.500000e-02 : f32,
       pool_size = 2 : i64, pool_stride = 2 : i64}
      : (memref<1x14x14x16xi8>, memref<3x3x16x16xi8>, memref<1x7x7x16xi8>)
  return
}

// Depthwise: one filter per channel, kernel laid out (C, KH, KW).
// CHECK-LABEL: define void @dwconv
// CHECK:         call void @tiled_conv_dw_auto(i32 1, i32 14, i32 14, i32 16, i32 14, i32 14, i32 1, i32 1, i32 3,
func.func @dwconv(%in: memref<1x14x14x16xi8>, %f: memref<16x3x3xi8>,
                  %b: memref<16xi32>, %out: memref<1x14x14x16xi8>) {
  gemmlir.depthwise_conv2d_i8(%in, %f, %out) bias(%b : memref<16xi32>)
      {padding = 1 : i64, scale = 2.500000e-02 : f32}
      : (memref<1x14x14x16xi8>, memref<16x3x3xi8>, memref<1x14x14x16xi8>)
  return
}

// BAD: error: 'gemmlir.conv2d_i8' op output extent on axis 0 is 12, expected 14

