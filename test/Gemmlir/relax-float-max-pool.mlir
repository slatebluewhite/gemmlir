// RUN: gemmlir-opt %s --relax-float-max-pool --split-input-file | FileCheck %s

// A float max-pool whose result only becomes an integer: the IEEE maximum
// becomes `arith.maxnumf`, which is one `fmax.s`.

// CHECK-LABEL: func.func @pool_into_quantize
// CHECK-NOT:     linalg.pooling_nhwc_max
// CHECK:         linalg.generic
// CHECK:           arith.maxnumf
func.func @pool_into_quantize(%src: memref<1x8x8x4xf32>, %out: memref<1x4x4x4xi8>) {
  %win = memref.alloc() : memref<2x2xf32>
  %pool = memref.alloc() : memref<1x4x4x4xf32>
  linalg.pooling_nhwc_max {strides = dense<2> : tensor<2xi64>,
                           dilations = dense<1> : tensor<2xi64>}
    ins(%src, %win : memref<1x8x8x4xf32>, memref<2x2xf32>)
    outs(%pool : memref<1x4x4x4xf32>)
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%pool : memref<1x4x4x4xf32>) outs(%out : memref<1x4x4x4xi8>) {
  ^bb0(%in: f32, %o: i8):
    %q = arith.fptosi %in : f32 to i32
    %t = arith.trunci %q : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %pool : memref<1x4x4x4xf32>
  memref.dealloc %win : memref<2x2xf32>
  return
}

// -----

// The result is also read as a float, so the NaN behaviour is observable.

// CHECK-LABEL: func.func @pool_into_float
// CHECK:         linalg.pooling_nhwc_max
func.func @pool_into_float(%src: memref<1x8x8x4xf32>, %out: memref<1x4x4x4xf32>) {
  %win = memref.alloc() : memref<2x2xf32>
  %pool = memref.alloc() : memref<1x4x4x4xf32>
  linalg.pooling_nhwc_max {strides = dense<2> : tensor<2xi64>,
                           dilations = dense<1> : tensor<2xi64>}
    ins(%src, %win : memref<1x8x8x4xf32>, memref<2x2xf32>)
    outs(%pool : memref<1x4x4x4xf32>)
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%pool : memref<1x4x4x4xf32>) outs(%out : memref<1x4x4x4xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  }
  memref.dealloc %pool : memref<1x4x4x4xf32>
  memref.dealloc %win : memref<2x2xf32>
  return
}

// -----

// An integer pool is already one instruction; leave it alone.

// CHECK-LABEL: func.func @pool_i8
// CHECK:         linalg.pooling_nhwc_max
func.func @pool_i8(%src: memref<1x8x8x4xi8>, %out: memref<1x4x4x4xi8>) {
  %win = memref.alloc() : memref<2x2xi8>
  linalg.pooling_nhwc_max {strides = dense<2> : tensor<2xi64>,
                           dilations = dense<1> : tensor<2xi64>}
    ins(%src, %win : memref<1x8x8x4xi8>, memref<2x2xi8>)
    outs(%out : memref<1x4x4x4xi8>)
  memref.dealloc %win : memref<2x2xi8>
  return
}

// -----

// The stride and dilation have to survive the rewrite.

// The maps are printed above the function, so they are checked before the label.
// CHECK:       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 3 + d4 * 2, d2 * 3 + d5 * 2, d3)>
// CHECK-LABEL: func.func @pool_strided
// CHECK:         arith.maxnumf
func.func @pool_strided(%src: memref<1x16x16x4xf32>, %out: memref<1x4x4x4xi8>) {
  %win = memref.alloc() : memref<2x2xf32>
  %pool = memref.alloc() : memref<1x4x4x4xf32>
  linalg.pooling_nhwc_max {strides = dense<3> : tensor<2xi64>,
                           dilations = dense<2> : tensor<2xi64>}
    ins(%src, %win : memref<1x16x16x4xf32>, memref<2x2xf32>)
    outs(%pool : memref<1x4x4x4xf32>)
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%pool : memref<1x4x4x4xf32>) outs(%out : memref<1x4x4x4xi8>) {
  ^bb0(%in: f32, %o: i8):
    %q = arith.fptosi %in : f32 to i32
    %t = arith.trunci %q : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %pool : memref<1x4x4x4xf32>
  memref.dealloc %win : memref<2x2xf32>
  return
}

// -----

// A relu below the pool is `arith.maxnumf`, which **erases** a NaN instead of
// carrying it to the conversion -- so the two maxima differ by a defined
// number, not by poison, and the rewrite is refused.

// CHECK-LABEL: func.func @pool_into_relu
// CHECK:         linalg.pooling_nhwc_max
func.func @pool_into_relu(%src: memref<1x8x8x4xf32>, %out: memref<1x4x4x4xi8>) {
  %zero = arith.constant 0.0 : f32
  %win = memref.alloc() : memref<2x2xf32>
  %pool = memref.alloc() : memref<1x4x4x4xf32>
  linalg.pooling_nhwc_max {strides = dense<2> : tensor<2xi64>,
                           dilations = dense<1> : tensor<2xi64>}
    ins(%src, %win : memref<1x8x8x4xf32>, memref<2x2xf32>)
    outs(%pool : memref<1x4x4x4xf32>)
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%pool : memref<1x4x4x4xf32>) outs(%out : memref<1x4x4x4xi8>) {
  ^bb0(%in: f32, %o: i8):
    %r = arith.maxnumf %in, %zero : f32
    %q = arith.fptosi %r : f32 to i32
    %t = arith.trunci %q : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %pool : memref<1x4x4x4xf32>
  memref.dealloc %win : memref<2x2xf32>
  return
}

// -----

// A comparison answers `false` for a NaN and something else for a number; that
// answer is defined, so the difference is observable even though it is an i1.

// CHECK-LABEL: func.func @pool_into_compare
// CHECK:         linalg.pooling_nhwc_max
func.func @pool_into_compare(%src: memref<1x8x8x4xf32>, %out: memref<1x4x4x4xi8>) {
  %k = arith.constant 1.0 : f32
  %lo = arith.constant 0.0 : f32
  %win = memref.alloc() : memref<2x2xf32>
  %pool = memref.alloc() : memref<1x4x4x4xf32>
  linalg.pooling_nhwc_max {strides = dense<2> : tensor<2xi64>,
                           dilations = dense<1> : tensor<2xi64>}
    ins(%src, %win : memref<1x8x8x4xf32>, memref<2x2xf32>)
    outs(%pool : memref<1x4x4x4xf32>)
  linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%pool : memref<1x4x4x4xf32>) outs(%out : memref<1x4x4x4xi8>) {
  ^bb0(%in: f32, %o: i8):
    %c = arith.cmpf olt, %in, %k : f32
    %s = arith.select %c, %lo, %in : f32
    %q = arith.fptosi %s : f32 to i32
    %t = arith.trunci %q : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %pool : memref<1x4x4x4xf32>
  memref.dealloc %win : memref<2x2xf32>
  return
}
