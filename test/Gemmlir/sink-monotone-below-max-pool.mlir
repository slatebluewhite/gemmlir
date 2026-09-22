// RUN: gemmlir-opt --sink-monotone-below-max-pool --split-input-file %s | FileCheck %s

// DenseNet's stem dequantizes its first convolution into f32, pads that, and
// max-pools it -- so the dequantization runs on four times as many elements as
// anything downstream reads, through two buffers walked once each.
//
// `max` commutes with a monotone non-decreasing map. Measured as the stem's two
// loops at DenseNet's own shape: -33.6%, every output identical.

#acc = affine_map<(d0, d1, d2, d3) -> (0, d2, d3, d1)>
#chan = affine_map<(d0, d1, d2, d3) -> (d1)>
#out = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @stem
// The f32 buffers are gone and the pool runs in the accumulator's own type:
// CHECK-NOT:   memref.alloc() {{.*}} memref<1x8x8x4xf32>
// CHECK-NOT:   memref.alloc() {{.*}} memref<1x10x10x4xf32>
// CHECK-DAG:   %[[N:.*]] = arith.constant -2147483648 : i32
// CHECK-DAG:   %[[PAD:.*]] = memref.alloc() {{.*}} memref<1x10x10x4xi32>
// CHECK:       linalg.fill ins(%[[N]]{{.*}} outs(%[[PAD]]
// CHECK:       memref.subview %[[PAD]][0, 1, 1, 0] [1, 8, 8, 4] [1, 1, 1, 1]
// CHECK:       memref.copy %arg0
// CHECK:       linalg.pooling_nhwc_max
// and the dequantization runs once, on what the pool left:
// CHECK:       linalg.generic
// CHECK:         arith.sitofp
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:         arith.cmpf
func.func @stem(%acc: memref<1x8x8x4xi32>, %bias: memref<4xf32>,
                %out: memref<1x4x4x4xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %scale = arith.constant 1.250000e-01 : f32
  %wide = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x4xf32>
  linalg.generic {indexing_maps = [#acc, #chan, #out], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc, %bias : memref<1x8x8x4xi32>, memref<4xf32>)
      outs(%wide : memref<1x8x8x4xf32>) {
  ^bb0(%in: i32, %b: f32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %scale : f32
    %p = arith.addf %m, %b : f32
    %g = arith.cmpf ugt, %p, %zero : f32
    %r = arith.select %g, %p, %zero : f32
    linalg.yield %r : f32
  }
  %win = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
  %pad = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x4xf32>
  linalg.map outs(%pad : memref<1x10x10x4xf32>)
    (%init: f32) {
      linalg.yield %zero : f32
    }
  %inner = memref.subview %pad[0, 1, 1, 0] [1, 8, 8, 4] [1, 1, 1, 1] : memref<1x10x10x4xf32> to memref<1x8x8x4xf32, strided<[400, 40, 4, 1], offset: 44>>
  memref.copy %wide, %inner : memref<1x8x8x4xf32> to memref<1x8x8x4xf32, strided<[400, 40, 4, 1], offset: 44>>
  linalg.fill ins(%ninf : f32) outs(%out : memref<1x4x4x4xf32>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%pad, %win : memref<1x10x10x4xf32>, memref<3x3xf32>)
      outs(%out : memref<1x4x4x4xf32>)
  memref.dealloc %pad : memref<1x10x10x4xf32>
  memref.dealloc %wide : memref<1x8x8x4xf32>
  return
}

// -----

// A negative scale is not monotone increasing, so the maximum is not the same
// element and nothing may move.

#acc = affine_map<(d0, d1, d2, d3) -> (0, d2, d3, d1)>
#chan = affine_map<(d0, d1, d2, d3) -> (d1)>
#out = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @negative_scale_refused
// CHECK:       memref.alloc() {{.*}} memref<1x8x8x4xf32>
// CHECK:       linalg.pooling_nhwc_max
// CHECK-SAME:    memref<1x10x10x4xf32>
func.func @negative_scale_refused(%acc: memref<1x8x8x4xi32>, %bias: memref<4xf32>,
                                  %out: memref<1x4x4x4xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %scale = arith.constant -1.250000e-01 : f32
  %wide = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x4xf32>
  linalg.generic {indexing_maps = [#acc, #chan, #out], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc, %bias : memref<1x8x8x4xi32>, memref<4xf32>)
      outs(%wide : memref<1x8x8x4xf32>) {
  ^bb0(%in: i32, %b: f32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %scale : f32
    %p = arith.addf %m, %b : f32
    %g = arith.cmpf ugt, %p, %zero : f32
    %r = arith.select %g, %p, %zero : f32
    linalg.yield %r : f32
  }
  %win = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
  %pad = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x4xf32>
  linalg.map outs(%pad : memref<1x10x10x4xf32>)
    (%init: f32) {
      linalg.yield %zero : f32
    }
  %inner = memref.subview %pad[0, 1, 1, 0] [1, 8, 8, 4] [1, 1, 1, 1] : memref<1x10x10x4xf32> to memref<1x8x8x4xf32, strided<[400, 40, 4, 1], offset: 44>>
  memref.copy %wide, %inner : memref<1x8x8x4xf32> to memref<1x8x8x4xf32, strided<[400, 40, 4, 1], offset: 44>>
  linalg.fill ins(%ninf : f32) outs(%out : memref<1x4x4x4xf32>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%pad, %win : memref<1x10x10x4xf32>, memref<3x3xf32>)
      outs(%out : memref<1x4x4x4xf32>)
  memref.dealloc %pad : memref<1x10x10x4xf32>
  memref.dealloc %wide : memref<1x8x8x4xf32>
  return
}

// -----

// A pad that is not the map's own floor would read back differently, because
// the smallest integer comes out of the map at the floor and nowhere else.

#acc = affine_map<(d0, d1, d2, d3) -> (0, d2, d3, d1)>
#chan = affine_map<(d0, d1, d2, d3) -> (d1)>
#out = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @wrong_pad_refused
// CHECK:       memref.alloc() {{.*}} memref<1x8x8x4xf32>
// CHECK:       linalg.pooling_nhwc_max
// CHECK-SAME:    memref<1x10x10x4xf32>
func.func @wrong_pad_refused(%acc: memref<1x8x8x4xi32>, %bias: memref<4xf32>,
                             %out: memref<1x4x4x4xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  %five = arith.constant 5.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %scale = arith.constant 1.250000e-01 : f32
  %wide = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x4xf32>
  linalg.generic {indexing_maps = [#acc, #chan, #out], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc, %bias : memref<1x8x8x4xi32>, memref<4xf32>)
      outs(%wide : memref<1x8x8x4xf32>) {
  ^bb0(%in: i32, %b: f32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %scale : f32
    %p = arith.addf %m, %b : f32
    %g = arith.cmpf ugt, %p, %zero : f32
    %r = arith.select %g, %p, %zero : f32
    linalg.yield %r : f32
  }
  %win = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
  %pad = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x4xf32>
  linalg.map outs(%pad : memref<1x10x10x4xf32>)
    (%init: f32) {
      linalg.yield %five : f32
    }
  %inner = memref.subview %pad[0, 1, 1, 0] [1, 8, 8, 4] [1, 1, 1, 1] : memref<1x10x10x4xf32> to memref<1x8x8x4xf32, strided<[400, 40, 4, 1], offset: 44>>
  memref.copy %wide, %inner : memref<1x8x8x4xf32> to memref<1x8x8x4xf32, strided<[400, 40, 4, 1], offset: 44>>
  linalg.fill ins(%ninf : f32) outs(%out : memref<1x4x4x4xf32>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%pad, %win : memref<1x10x10x4xf32>, memref<3x3xf32>)
      outs(%out : memref<1x4x4x4xf32>)
  memref.dealloc %pad : memref<1x10x10x4xf32>
  memref.dealloc %wide : memref<1x8x8x4xf32>
  return
}
