// A convolution's padding is a `tensor.pad` in linalg, and bufferizing it leaves
// an allocation, a fill of the whole of it, and a copy of the real input into
// the middle -- 972 zero stores and a strided 768-element copy on the CNN, for
// an input of 768 elements. `tiled_conv_auto` takes a `padding` and does it
// while it reads the image, so all of that goes away: 2.15 ms to 1.12 ms on the
// board, with the runtime's own CPU implementation agreeing to the digit.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @folds_padding
// CHECK-NOT:     memref.alloc() {{.*}} memref<1x18x18x3xi8>
// CHECK-NOT:     memref.copy
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %arg3)
// CHECK-SAME:      bias(%arg2 : memref<8xi32>)
// CHECK-SAME:      {act = #gemmlir.act<relu>, padding = 1 : i64
// CHECK-SAME:      (memref<1x16x16x3xi8>, memref<3x3x3x8xi8>, memref<1x16x16x8xi8>)
func.func @folds_padding(%in: memref<1x16x16x3xi8>, %f: memref<3x3x3x8xi8>,
                         %b: memref<8xi32>, %out: memref<1x16x16x8xi8>) {
  %z = arith.constant 0 : i8
  %p = memref.alloc() : memref<1x18x18x3xi8>
  linalg.fill ins(%z : i8) outs(%p : memref<1x18x18x3xi8>)
  %w = memref.subview %p[0, 1, 1, 0] [1, 16, 16, 3] [1, 1, 1, 1]
     : memref<1x18x18x3xi8> to memref<1x16x16x3xi8, strided<[972, 54, 3, 1], offset: 57>>
  memref.copy %in, %w : memref<1x16x16x3xi8> to memref<1x16x16x3xi8, strided<[972, 54, 3, 1], offset: 57>>
  gemmlir.conv2d_i8(%p, %f, %out) bias(%b : memref<8xi32>)
    {act = #gemmlir.act<relu>, scale = 2.000000e-02 : f32}
    : (memref<1x18x18x3xi8>, memref<3x3x3x8xi8>, memref<1x16x16x8xi8>)
  memref.dealloc %p : memref<1x18x18x3xi8>
  return
}

// The runtime has one `padding` scalar, so it cannot express a window that is
// not centred. Left alone.
// CHECK-LABEL: func.func @asymmetric_stays
// CHECK:         memref.copy
// CHECK:         gemmlir.conv2d_i8(%alloc
func.func @asymmetric_stays(%in: memref<1x16x16x3xi8>, %f: memref<3x3x3x8xi8>,
                            %b: memref<8xi32>, %out: memref<1x16x16x8xi8>) {
  %z = arith.constant 0 : i8
  %p = memref.alloc() : memref<1x18x18x3xi8>
  linalg.fill ins(%z : i8) outs(%p : memref<1x18x18x3xi8>)
  %w = memref.subview %p[0, 0, 1, 0] [1, 16, 16, 3] [1, 1, 1, 1]
     : memref<1x18x18x3xi8> to memref<1x16x16x3xi8, strided<[972, 54, 3, 1], offset: 3>>
  memref.copy %in, %w : memref<1x16x16x3xi8> to memref<1x16x16x3xi8, strided<[972, 54, 3, 1], offset: 3>>
  gemmlir.conv2d_i8(%p, %f, %out) bias(%b : memref<8xi32>)
    {act = #gemmlir.act<relu>, scale = 2.000000e-02 : f32}
    : (memref<1x18x18x3xi8>, memref<3x3x3x8xi8>, memref<1x16x16x8xi8>)
  memref.dealloc %p : memref<1x18x18x3xi8>
  return
}

// Padding the channel axis is not padding the image. Left alone.
// CHECK-LABEL: func.func @channel_padding_stays
// CHECK:         memref.copy
// CHECK:         gemmlir.conv2d_i8(%alloc
func.func @channel_padding_stays(%in: memref<1x18x18x1xi8>, %f: memref<3x3x3x8xi8>,
                                 %b: memref<8xi32>, %out: memref<1x16x16x8xi8>) {
  %z = arith.constant 0 : i8
  %p = memref.alloc() : memref<1x18x18x3xi8>
  linalg.fill ins(%z : i8) outs(%p : memref<1x18x18x3xi8>)
  %w = memref.subview %p[0, 0, 0, 1] [1, 18, 18, 1] [1, 1, 1, 1]
     : memref<1x18x18x3xi8> to memref<1x18x18x1xi8, strided<[972, 54, 3, 1], offset: 1>>
  memref.copy %in, %w : memref<1x18x18x1xi8> to memref<1x18x18x1xi8, strided<[972, 54, 3, 1], offset: 1>>
  gemmlir.conv2d_i8(%p, %f, %out) bias(%b : memref<8xi32>)
    {act = #gemmlir.act<relu>, scale = 2.000000e-02 : f32}
    : (memref<1x18x18x3xi8>, memref<3x3x3x8xi8>, memref<1x16x16x8xi8>)
  memref.dealloc %p : memref<1x18x18x3xi8>
  return
}

// The hardware pads with zeros, so a border of anything else is a different
// convolution. Left alone.
// CHECK-LABEL: func.func @nonzero_border_stays
// CHECK:         memref.copy
// CHECK:         gemmlir.conv2d_i8(%alloc
func.func @nonzero_border_stays(%in: memref<1x16x16x3xi8>, %f: memref<3x3x3x8xi8>,
                                %b: memref<8xi32>, %out: memref<1x16x16x8xi8>) {
  %one = arith.constant 1 : i8
  %p = memref.alloc() : memref<1x18x18x3xi8>
  linalg.fill ins(%one : i8) outs(%p : memref<1x18x18x3xi8>)
  %w = memref.subview %p[0, 1, 1, 0] [1, 16, 16, 3] [1, 1, 1, 1]
     : memref<1x18x18x3xi8> to memref<1x16x16x3xi8, strided<[972, 54, 3, 1], offset: 57>>
  memref.copy %in, %w : memref<1x16x16x3xi8> to memref<1x16x16x3xi8, strided<[972, 54, 3, 1], offset: 57>>
  gemmlir.conv2d_i8(%p, %f, %out) bias(%b : memref<8xi32>)
    {act = #gemmlir.act<relu>, scale = 2.000000e-02 : f32}
    : (memref<1x18x18x3xi8>, memref<3x3x3x8xi8>, memref<1x16x16x8xi8>)
  memref.dealloc %p : memref<1x18x18x3xi8>
  return
}

// The depthwise call takes a `padding` too, so the same fold applies -- on the
// MobileNet-shaped model it was 2592 of the 3370 elements left, 3.43 ms to
// 1.12 ms.
// CHECK-LABEL: func.func @folds_depthwise_padding
// CHECK-NOT:     memref.copy
// CHECK:         gemmlir.depthwise_conv2d_i8(%arg0, %arg1, %arg3)
// CHECK-SAME:      {act = #gemmlir.act<relu>, padding = 1 : i64
// CHECK-SAME:      (memref<1x16x16x8xi8>, memref<8x3x3xi8>, memref<1x16x16x8xi8>)
func.func @folds_depthwise_padding(%in: memref<1x16x16x8xi8>, %f: memref<8x3x3xi8>,
                                   %b: memref<8xi32>, %out: memref<1x16x16x8xi8>) {
  %z = arith.constant 0 : i8
  %p = memref.alloc() : memref<1x18x18x8xi8>
  linalg.fill ins(%z : i8) outs(%p : memref<1x18x18x8xi8>)
  %w = memref.subview %p[0, 1, 1, 0] [1, 16, 16, 8] [1, 1, 1, 1]
     : memref<1x18x18x8xi8> to memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>
  memref.copy %in, %w : memref<1x16x16x8xi8> to memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>
  gemmlir.depthwise_conv2d_i8(%p, %f, %out) bias(%b : memref<8xi32>)
    {act = #gemmlir.act<relu>, scale = 2.000000e-02 : f32}
    : (memref<1x18x18x8xi8>, memref<8x3x3xi8>, memref<1x16x16x8xi8>)
  memref.dealloc %p : memref<1x18x18x8xi8>
  return
}

// A dilated convolution's shape-preserving padding is `dilation * (K-1) / 2`,
// which is 4 for a 3-tap filter at rate 4. `tiled_conv_auto` compares the
// padding against the *undilated* kernel -- `if (kernel_dim <= padding) {
// printf("kernel_dim must be larger than padding\n"); exit(1); }` -- and so
// refuses it, even though the dilated filter is 9 taps wide. Folding it in
// would produce a call the runtime exits on, so the border stays where it is:
// an explicit zero buffer and a convolution with no padding of its own, which
// is what this pattern started from and computes the same thing.
// CHECK-LABEL: func.func @dilated_padding_stays
// CHECK:         memref.copy
// CHECK:         gemmlir.conv2d_i8(%alloc
// CHECK-SAME:      dilation = 4 : i64
// CHECK-NOT:     padding
func.func @dilated_padding_stays(%in: memref<1x16x16x3xi8>, %f: memref<3x3x3x8xi8>,
                                 %b: memref<8xi32>, %out: memref<1x16x16x8xi8>) {
  %z = arith.constant 0 : i8
  %p = memref.alloc() : memref<1x24x24x3xi8>
  linalg.fill ins(%z : i8) outs(%p : memref<1x24x24x3xi8>)
  %w = memref.subview %p[0, 4, 4, 0] [1, 16, 16, 3] [1, 1, 1, 1]
     : memref<1x24x24x3xi8> to memref<1x16x16x3xi8, strided<[1728, 72, 3, 1], offset: 300>>
  memref.copy %in, %w : memref<1x16x16x3xi8> to memref<1x16x16x3xi8, strided<[1728, 72, 3, 1], offset: 300>>
  gemmlir.conv2d_i8(%p, %f, %out) bias(%b : memref<8xi32>)
    {act = #gemmlir.act<relu>, dilation = 4 : i64, scale = 2.000000e-02 : f32}
    : (memref<1x24x24x3xi8>, memref<3x3x3x8xi8>, memref<1x16x16x8xi8>)
  memref.dealloc %p : memref<1x24x24x3xi8>
  return
}

// A rate-2 filter's padding is 2, which is still inside a 3-tap kernel, so it
// folds like any other.
// CHECK-LABEL: func.func @rate_two_folds
// CHECK-NOT:     memref.copy
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %arg3)
// CHECK-SAME:      dilation = 2 : i64, padding = 2 : i64
func.func @rate_two_folds(%in: memref<1x16x16x3xi8>, %f: memref<3x3x3x8xi8>,
                          %b: memref<8xi32>, %out: memref<1x16x16x8xi8>) {
  %z = arith.constant 0 : i8
  %p = memref.alloc() : memref<1x20x20x3xi8>
  linalg.fill ins(%z : i8) outs(%p : memref<1x20x20x3xi8>)
  %w = memref.subview %p[0, 2, 2, 0] [1, 16, 16, 3] [1, 1, 1, 1]
     : memref<1x20x20x3xi8> to memref<1x16x16x3xi8, strided<[1200, 60, 3, 1], offset: 126>>
  memref.copy %in, %w : memref<1x16x16x3xi8> to memref<1x16x16x3xi8, strided<[1200, 60, 3, 1], offset: 126>>
  gemmlir.conv2d_i8(%p, %f, %out) bias(%b : memref<8xi32>)
    {act = #gemmlir.act<relu>, dilation = 2 : i64, scale = 2.000000e-02 : f32}
    : (memref<1x20x20x3xi8>, memref<3x3x3x8xi8>, memref<1x16x16x8xi8>)
  memref.dealloc %p : memref<1x20x20x3xi8>
  return
}

// The matcher declines the same thing, so such a convolution is left as a loop
// rather than becoming an operation that will not verify. Bufferizing a padded
// convolution whose input is already i8 writes the result straight into the
// middle of the padded buffer, which is a 16x16 window of an 18x18 one.
// CHECK-LABEL: func.func @window_narrower_than_its_buffer_stays_a_loop
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:     gemmlir.conv2d_i8
func.func @window_narrower_than_its_buffer_stays_a_loop(
    %in: memref<1x18x18x8xi8>, %f: memref<3x3x8x8xi8>, %pad: memref<1x18x18x8xi8>) {
  %z = arith.constant 0 : i32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<1x16x16x8xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<1x16x16x8xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %f : memref<1x18x18x8xi8>, memref<3x3x8x8xi8>)
    outs(%acc : memref<1x16x16x8xi32>)
  %w = memref.subview %pad[0, 1, 1, 0] [1, 16, 16, 8] [1, 1, 1, 1]
     : memref<1x18x18x8xi8> to memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%acc : memref<1x16x16x8xi32>)
    outs(%w : memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>) {
  ^bb0(%a: i32, %b: i8):
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    %r = arith.fptosi %m : f32 to i32
    %c1 = arith.maxsi %r, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<1x16x16x8xi32>
  return
}
