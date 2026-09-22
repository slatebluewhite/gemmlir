// RUN: gemmlir-opt %s --depthwise-as-block-diagonal --split-input-file | FileCheck %s

// `tiled_conv_dw_auto` sets `pochs` to 1, so a depthwise layer costs one
// accelerator call per channel. The array is 16x16 and a depthwise uses one
// column of it; filling the other fifteen with zeros costs nothing there and
// turns sixteen calls into one.

// CHECK-LABEL: memref.global "private" constant @f_blockdiag : memref<2x3x3x16x16xi8>
// CHECK-LABEL: func.func @two_groups
// CHECK-NOT:     gemmlir.depthwise_conv2d_i8
// CHECK:         %[[B:.*]] = memref.get_global @f_blockdiag
// CHECK:         memref.subview %arg0[0, 0, 0, 0] [1, 8, 8, 16]
// CHECK:         memref.subview %[[B]][0, 0, 0, 0, 0] [1, 3, 3, 16, 16]
// CHECK:         gemmlir.conv2d_i8
// CHECK:         memref.subview %arg0[0, 0, 0, 16] [1, 8, 8, 16]
// CHECK:         memref.subview %[[B]][1, 0, 0, 0, 0] [1, 3, 3, 16, 16]
// CHECK:         gemmlir.conv2d_i8
memref.global "private" constant @f : memref<32x3x3xi8> = dense<2>
func.func @two_groups(%in: memref<1x8x8x32xi8>, %out: memref<1x8x8x32xi8>,
                      %b: memref<32xi32>) {
  %f = memref.get_global @f : memref<32x3x3xi8>
  gemmlir.depthwise_conv2d_i8(%in, %f, %out) bias(%b : memref<32xi32>)
    {padding = 1 : i64, stride = 1 : i64, scale = 5.0e-01 : f32}
    : (memref<1x8x8x32xi8>, memref<32x3x3xi8>, memref<1x8x8x32xi8>)
  return
}

// -----

// The weight goes on the diagonal of its group and nowhere else, so no channel
// can reach another: off the diagonal it is multiplied by zero. Shown at two
// lanes, where the constant prints as itself.
// RUN: gemmlir-opt %s --depthwise-as-block-diagonal="lanes=2" --split-input-file \
// RUN:   | FileCheck %s --check-prefix=TWO

// TWO: memref.global "private" constant @g_blockdiag : memref<2x1x1x2x2xi8>
// TWO-SAME: dense<{{\[}}{{\[}}{{\[}}{{\[}}{{\[}}1, 0], [0, 2]]]], {{\[}}{{\[}}{{\[}}{{\[}}3, 0], [0, 4]]]]]>
// TWO-LABEL: func.func @the_diagonal
// TWO-COUNT-2: gemmlir.conv2d_i8
memref.global "private" constant @g : memref<4x1x1xi8> =
  dense<[[[1]], [[2]], [[3]], [[4]]]>
func.func @the_diagonal(%in: memref<1x4x4x4xi8>, %out: memref<1x4x4x4xi8>) {
  %g = memref.get_global @g : memref<4x1x1xi8>
  gemmlir.depthwise_conv2d_i8(%in, %g, %out) {stride = 1 : i64}
    : (memref<1x4x4x4xi8>, memref<4x1x1xi8>, memref<1x4x4x4xi8>)
  return
}

// -----

// A channel count that is not a multiple of `lanes` leaves a remainder, and
// that last group gets its **own** global: the lowering passes the filter's
// out-channel count as the weight stride, so a narrower group cannot be a slice
// of a wider one. MobileNetV3, MnasNet and ShuffleNet are 24, 72, 88 or 120
// channels -- a remainder of eight every time.

// CHECK-LABEL: memref.global "private" constant @h_blockdiag : memref<1x3x3x16x16xi8>
// CHECK-LABEL: memref.global "private" constant @h_blockdiag_tail : memref<3x3x8x8xi8>
// CHECK-LABEL: func.func @ragged_channels
// CHECK-NOT:     gemmlir.depthwise_conv2d_i8
// CHECK:         memref.subview %arg0[0, 0, 0, 0] [1, 8, 8, 16]
// CHECK:         gemmlir.conv2d_i8
// CHECK:         memref.subview %arg0[0, 0, 0, 16] [1, 8, 8, 8]
// CHECK:         gemmlir.conv2d_i8
memref.global "private" constant @h : memref<24x3x3xi8> = dense<2>
func.func @ragged_channels(%in: memref<1x8x8x24xi8>, %out: memref<1x8x8x24xi8>) {
  %h = memref.get_global @h : memref<24x3x3xi8>
  gemmlir.depthwise_conv2d_i8(%in, %h, %out) {padding = 1 : i64, stride = 1 : i64}
    : (memref<1x8x8x24xi8>, memref<24x3x3xi8>, memref<1x8x8x24xi8>)
  return
}

// -----

// Fewer channels than one group: nothing to gather.

// CHECK-LABEL: func.func @one_group_only
// CHECK:         gemmlir.depthwise_conv2d_i8
memref.global "private" constant @k : memref<8x3x3xi8> = dense<2>
func.func @one_group_only(%in: memref<1x8x8x8xi8>, %out: memref<1x8x8x8xi8>) {
  %k = memref.get_global @k : memref<8x3x3xi8>
  gemmlir.depthwise_conv2d_i8(%in, %k, %out) {padding = 1 : i64, stride = 1 : i64}
    : (memref<1x8x8x8xi8>, memref<8x3x3xi8>, memref<1x8x8x8xi8>)
  return
}

// -----

// A filter this pass cannot read is left alone.

// CHECK-LABEL: func.func @filter_not_constant
// CHECK:         gemmlir.depthwise_conv2d_i8
func.func @filter_not_constant(%in: memref<1x8x8x32xi8>, %f: memref<32x3x3xi8>,
                               %out: memref<1x8x8x32xi8>) {
  gemmlir.depthwise_conv2d_i8(%in, %f, %out) {padding = 1 : i64, stride = 1 : i64}
    : (memref<1x8x8x32xi8>, memref<32x3x3xi8>, memref<1x8x8x32xi8>)
  return
}
