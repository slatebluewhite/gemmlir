// RUN: gemmlir-opt %s --pool-without-padding --split-input-file | FileCheck %s

// A `ceil_mode` padding on the high side only: the window overhangs on the very
// last output row and column and nowhere else, so the output is four bands --
// the interior with the full 3x3 window, two strips with 3x2 and 2x3, and the
// corner with 2x2. The padded buffer, its fill and the copy into it all go.

// CHECK-LABEL: func.func @high_only
// CHECK-NOT:     memref.copy
// CHECK-NOT:     memref<1x50x50x64xf32>
// CHECK:         %[[IN:.*]] = memref.subview %arg0[0, 0, 0, 0] [1, 47, 47, 64]
// CHECK:         %[[OUT:.*]] = memref.subview %arg1[0, 0, 0, 0] [1, 23, 23, 64]
// CHECK:         %[[W:.*]] = memref.subview %{{.*}}[0, 0] [3, 3] [1, 1]
// CHECK:         linalg.pooling_nhwc_max {{.*}} ins(%[[IN]], %[[W]]
// CHECK-SAME:      outs(%[[OUT]]
// CHECK:         memref.subview %arg0[0, 0, 46, 0] [1, 47, 2, 64]
// CHECK:         memref.subview %arg1[0, 0, 23, 0] [1, 23, 1, 64]
// CHECK:         memref.subview %{{.*}}[0, 0] [3, 2] [1, 1]
// CHECK:         linalg.fill ins(%[[P:.*]] : f32)
// CHECK:         linalg.pooling_nhwc_max
// CHECK:         memref.subview %arg0[0, 46, 0, 0] [1, 2, 47, 64]
// CHECK:         memref.subview %{{.*}}[0, 0] [2, 3] [1, 1]
// CHECK:         memref.subview %arg0[0, 46, 46, 0] [1, 2, 2, 64]
// CHECK:         memref.subview %{{.*}}[0, 0] [2, 2] [1, 1]
func.func @high_only(%src: memref<1x48x48x64xf32>, %out: memref<1x24x24x64xf32>,
                     %win: memref<3x3xf32>) {
  %p = arith.constant 0.0 : f32
  %lo = arith.constant 0xFF800000 : f32
  %pad = memref.alloc() : memref<1x50x50x64xf32>
  linalg.fill ins(%p : f32) outs(%pad : memref<1x50x50x64xf32>)
  %s = memref.subview %pad[0, 0, 0, 0] [1, 48, 48, 64] [1, 1, 1, 1]
    : memref<1x50x50x64xf32> to memref<1x48x48x64xf32, strided<[160000, 3200, 64, 1]>>
  memref.copy %src, %s : memref<1x48x48x64xf32> to memref<1x48x48x64xf32, strided<[160000, 3200, 64, 1]>>
  linalg.fill ins(%lo : f32) outs(%out : memref<1x24x24x64xf32>)
  linalg.pooling_nhwc_max {strides = dense<2> : vector<2xi64>,
                           dilations = dense<1> : vector<2xi64>}
    ins(%pad, %win : memref<1x50x50x64xf32>, memref<3x3xf32>)
    outs(%out : memref<1x24x24x64xf32>)
  memref.dealloc %pad : memref<1x50x50x64xf32>
  return
}

// -----

// Padding on both sides at stride one: the first output row is clipped at the
// top, the last at the bottom, and the six rows in between are whole -- three
// bands each way, nine regions.

// CHECK-LABEL: func.func @both_sides
// CHECK-NOT:     memref.copy
// CHECK-COUNT-9: linalg.pooling_nhwc_max
// CHECK-NOT:     linalg.pooling_nhwc_max
func.func @both_sides(%src: memref<1x6x6x32xf32>, %out: memref<1x6x6x32xf32>,
                      %win: memref<3x3xf32>) {
  %p = arith.constant 0.0 : f32
  %lo = arith.constant 0xFF800000 : f32
  %pad = memref.alloc() : memref<1x9x9x32xf32>
  linalg.fill ins(%p : f32) outs(%pad : memref<1x9x9x32xf32>)
  %s = memref.subview %pad[0, 1, 1, 0] [1, 6, 6, 32] [1, 1, 1, 1]
    : memref<1x9x9x32xf32> to memref<1x6x6x32xf32, strided<[2592, 288, 32, 1], offset: 320>>
  memref.copy %src, %s : memref<1x6x6x32xf32> to memref<1x6x6x32xf32, strided<[2592, 288, 32, 1], offset: 320>>
  linalg.fill ins(%lo : f32) outs(%out : memref<1x6x6x32xf32>)
  linalg.pooling_nhwc_max {strides = dense<1> : vector<2xi64>,
                           dilations = dense<1> : vector<2xi64>}
    ins(%pad, %win : memref<1x9x9x32xf32>, memref<3x3xf32>)
    outs(%out : memref<1x6x6x32xf32>)
  memref.dealloc %pad : memref<1x9x9x32xf32>
  return
}

// -----

// A pad the window never reaches is one band: the pool reads the real image
// through a subview and the padded buffer goes away entirely. This is the case
// `--drop-unread-padding` catches in the frontend; it is handled here too
// rather than assumed.

// CHECK-LABEL: func.func @never_reaches_the_padding
// CHECK-NOT:     memref.copy
// CHECK-COUNT-1: linalg.pooling_nhwc_max
// CHECK-NOT:     linalg.pooling_nhwc_max
func.func @never_reaches_the_padding(%src: memref<1x8x8x32xf32>,
                                     %out: memref<1x4x4x32xf32>,
                                     %win: memref<2x2xf32>) {
  %p = arith.constant 0.0 : f32
  %lo = arith.constant 0xFF800000 : f32
  %pad = memref.alloc() : memref<1x9x9x32xf32>
  linalg.fill ins(%p : f32) outs(%pad : memref<1x9x9x32xf32>)
  %s = memref.subview %pad[0, 0, 0, 0] [1, 8, 8, 32] [1, 1, 1, 1]
    : memref<1x9x9x32xf32> to memref<1x8x8x32xf32, strided<[2592, 288, 32, 1]>>
  memref.copy %src, %s : memref<1x8x8x32xf32> to memref<1x8x8x32xf32, strided<[2592, 288, 32, 1]>>
  linalg.fill ins(%lo : f32) outs(%out : memref<1x4x4x32xf32>)
  linalg.pooling_nhwc_max {strides = dense<2> : vector<2xi64>,
                           dilations = dense<1> : vector<2xi64>}
    ins(%pad, %win : memref<1x9x9x32xf32>, memref<2x2xf32>)
    outs(%out : memref<1x4x4x32xf32>)
  memref.dealloc %pad : memref<1x9x9x32xf32>
  return
}

// -----

// Something else reads the padded buffer, so it is not simply a padding.

// CHECK-LABEL: func.func @another_reader
// CHECK:         memref.copy
// CHECK:         linalg.pooling_nhwc_max
// CHECK-NOT:     linalg.pooling_nhwc_max
func.func @another_reader(%src: memref<1x48x48x64xf32>, %out: memref<1x24x24x64xf32>,
                          %win: memref<3x3xf32>, %spy: memref<1x50x50x64xf32>) {
  %p = arith.constant 0.0 : f32
  %lo = arith.constant 0xFF800000 : f32
  %pad = memref.alloc() : memref<1x50x50x64xf32>
  linalg.fill ins(%p : f32) outs(%pad : memref<1x50x50x64xf32>)
  %s = memref.subview %pad[0, 0, 0, 0] [1, 48, 48, 64] [1, 1, 1, 1]
    : memref<1x50x50x64xf32> to memref<1x48x48x64xf32, strided<[160000, 3200, 64, 1]>>
  memref.copy %src, %s : memref<1x48x48x64xf32> to memref<1x48x48x64xf32, strided<[160000, 3200, 64, 1]>>
  linalg.fill ins(%lo : f32) outs(%out : memref<1x24x24x64xf32>)
  linalg.pooling_nhwc_max {strides = dense<2> : vector<2xi64>,
                           dilations = dense<1> : vector<2xi64>}
    ins(%pad, %win : memref<1x50x50x64xf32>, memref<3x3xf32>)
    outs(%out : memref<1x24x24x64xf32>)
  memref.copy %pad, %spy : memref<1x50x50x64xf32> to memref<1x50x50x64xf32>
  memref.dealloc %pad : memref<1x50x50x64xf32>
  return
}

// -----

// An i8 pool whose channels come in eights is one `--pack-int8-max-pool` turns
// into eight channels an `i64`. This used to be refused: bands write strided
// subviews and the packing could not collapse them, so the pool fell back to a
// byte at a time -- +30.1% on GoogLeNet against the 5.7% the banding saves on a
// model whose pool is f32. The packing reads a band now, so both apply and the
// padded copy goes.

// CHECK-LABEL: func.func @the_packable_bands_too
// CHECK:         linalg.pooling_nhwc_max
// CHECK:         linalg.pooling_nhwc_max
func.func @the_packable_bands_too(%src: memref<1x48x48x64xi8>,
                                    %out: memref<1x24x24x64xi8>,
                                    %win: memref<3x3xf32>) {
  %p = arith.constant 0 : i8
  %lo = arith.constant -128 : i8
  %pad = memref.alloc() : memref<1x50x50x64xi8>
  linalg.fill ins(%p : i8) outs(%pad : memref<1x50x50x64xi8>)
  %s = memref.subview %pad[0, 0, 0, 0] [1, 48, 48, 64] [1, 1, 1, 1]
    : memref<1x50x50x64xi8> to memref<1x48x48x64xi8, strided<[160000, 3200, 64, 1]>>
  memref.copy %src, %s : memref<1x48x48x64xi8> to memref<1x48x48x64xi8, strided<[160000, 3200, 64, 1]>>
  linalg.fill ins(%lo : i8) outs(%out : memref<1x24x24x64xi8>)
  linalg.pooling_nhwc_max {strides = dense<2> : vector<2xi64>,
                           dilations = dense<1> : vector<2xi64>}
    ins(%pad, %win : memref<1x50x50x64xi8>, memref<3x3xf32>)
    outs(%out : memref<1x24x24x64xi8>)
  memref.dealloc %pad : memref<1x50x50x64xi8>
  return
}
