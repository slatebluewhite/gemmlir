// A max-pool on a convolution's result folds into the convolution's own pooling.
//
// `tiled_conv_auto` pools the requantized i8 outputs with a plain max over the
// window, which is what linalg.pooling_nhwc_max computes. Its out-of-bounds
// branch treats the padding as zero rather than -inf, so only pool_padding = 0
// is emitted -- linalg has no pooling padding either.

// It is **off by default**, and not because the rewrite is wrong -- the
// runtime's own CPU implementation of the folded call gives exactly the
// unfolded answer. Folding removes an intermediate buffer, and on the U280
// board that shifts the allocator enough that the convolution's output address
// starts alternating between two bins, which this hardware cannot take. Two
// hand-written `tiled_conv_auto` calls in a row reproduce that with no compiler
// involved; see docs/pipeline.md.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir=fuse-pooling=true %s | FileCheck %s
// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s --check-prefix=OFF

// OFF-LABEL: func.func @qconv_pool
// OFF:         gemmlir.conv2d_i8
// OFF-NOT:       pool_stride
// OFF:         linalg.pooling_nhwc_max

#nhwc = affine_map<(n,h,w,f)->(n,h,w,f)>
#chan = affine_map<(n,h,w,f)->(f)>

// Five linalg ops become one call.
// CHECK-LABEL: func.func @qconv_pool
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %arg4) bias(%arg2 : memref<16xi32>)
// CHECK-SAME:    act = #gemmlir.act<relu>
// CHECK-SAME:    pool_size = 2 : i64, pool_stride = 2 : i64
// CHECK-NOT:     linalg.pooling_nhwc_max
// CHECK-NOT:     linalg.conv_2d_nhwc_hwcf
func.func @qconv_pool(%in: memref<1x16x16x16xi8>, %flt: memref<3x3x16x16xi8>,
                      %b: memref<16xi32>, %w: memref<2x2xi8>, %out: memref<1x7x7x16xi8>) {
  %z = arith.constant 0 : i32
  %s = arith.constant 2.500000e-02 : f32
  %lo = arith.constant 0 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<1x14x14x16xi32>
  %pre = memref.alloc() : memref<1x14x14x16xi8>
  linalg.fill ins(%z : i32) outs(%acc : memref<1x14x14x16xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt : memref<1x16x16x16xi8>, memref<3x3x16x16xi8>) outs(%acc : memref<1x14x14x16xi32>)
  linalg.generic {indexing_maps = [#chan, #nhwc], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%b : memref<16xi32>) outs(%acc : memref<1x14x14x16xi32>) {
  ^bb0(%bb: i32, %a: i32):
    %t = arith.addi %a, %bb : i32
    linalg.yield %t : i32
  }
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc : memref<1x14x14x16xi32>) outs(%pre : memref<1x14x14x16xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %x = arith.maxsi %i, %lo : i32
    %y = arith.minsi %x, %hi : i32
    %t = arith.trunci %y : i32 to i8
    linalg.yield %t : i8
  }
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%pre, %w : memref<1x14x14x16xi8>, memref<2x2xi8>) outs(%out : memref<1x7x7x16xi8>)
  memref.dealloc %acc : memref<1x14x14x16xi32>
  memref.dealloc %pre : memref<1x14x14x16xi8>
  return
}

// A pool whose window is not square, or whose strides differ between the axes,
// cannot be described by the runtime's three ints, so it stays put.
// CHECK-LABEL: func.func @uneven_pool
// CHECK:         gemmlir.conv2d_i8
// CHECK-NOT:     pool_size
// CHECK:         linalg.pooling_nhwc_max
func.func @uneven_pool(%in: memref<1x16x16x16xi8>, %flt: memref<3x3x16x16xi8>,
                       %w: memref<2x3xi8>, %out: memref<1x7x4x16xi8>) {
  %z = arith.constant 0 : i32
  %s = arith.constant 2.500000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<1x14x14x16xi32>
  %pre = memref.alloc() : memref<1x14x14x16xi8>
  linalg.fill ins(%z : i32) outs(%acc : memref<1x14x14x16xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt : memref<1x16x16x16xi8>, memref<3x3x16x16xi8>) outs(%acc : memref<1x14x14x16xi32>)
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc : memref<1x14x14x16xi32>) outs(%pre : memref<1x14x14x16xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %x = arith.maxsi %i, %lo : i32
    %y = arith.minsi %x, %hi : i32
    %t = arith.trunci %y : i32 to i8
    linalg.yield %t : i8
  }
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<[2, 3]> : tensor<2xi64>}
    ins(%pre, %w : memref<1x14x14x16xi8>, memref<2x3xi8>) outs(%out : memref<1x7x4x16xi8>)
  memref.dealloc %acc : memref<1x14x14x16xi32>
  memref.dealloc %pre : memref<1x14x14x16xi8>
  return
}

// A **padded** max-pool reaches here as a zero-filled buffer with the
// convolution's output copied into the middle of it -- which is exactly what
// `pool_padding` computes, since `sp_tiled_conv` reads zero out of bounds. The
// padding value is already zero by this point (`ZeroPadAMaxPool` normalises it
// back where the relu that makes -inf and zero agree is still visible), so
// there is nothing left to argue: the fill, the copy and the pool all go.
//
// Written against `gemmlir.conv2d_i8` because that is what the pattern matches
// -- by the time a padded pool is next to a convolution, the convolution has
// been converted.
// CHECK-LABEL: func.func @padded_pool_folds
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %arg3)
// CHECK-SAME:    pool_padding = 1 : i64, pool_size = 3 : i64, pool_stride = 2 : i64
// CHECK-NOT:     linalg.pooling_nhwc_max
// CHECK-NOT:     memref.copy
func.func @padded_pool_folds(%in: memref<1x8x8x16xi8>, %flt: memref<1x1x16x16xi8>,
                             %w: memref<3x3xi8>, %out: memref<1x4x4x16xi8>) {
  %zi8 = arith.constant 0 : i8
  %min = arith.constant -128 : i8
  %q = memref.alloc() : memref<1x8x8x16xi8>
  gemmlir.conv2d_i8(%in, %flt, %q) {act = #gemmlir.act<relu>, scale = 2.500000e-02 : f32}
    : (memref<1x8x8x16xi8>, memref<1x1x16x16xi8>, memref<1x8x8x16xi8>)
  %padded = memref.alloc() : memref<1x10x10x16xi8>
  linalg.fill ins(%zi8 : i8) outs(%padded : memref<1x10x10x16xi8>)
  %mid = memref.subview %padded[0, 1, 1, 0] [1, 8, 8, 16] [1, 1, 1, 1]
       : memref<1x10x10x16xi8> to memref<1x8x8x16xi8, strided<[1600, 160, 16, 1], offset: 176>>
  memref.copy %q, %mid : memref<1x8x8x16xi8> to memref<1x8x8x16xi8, strided<[1600, 160, 16, 1], offset: 176>>
  linalg.fill ins(%min : i8) outs(%out : memref<1x4x4x16xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%padded, %w : memref<1x10x10x16xi8>, memref<3x3xi8>) outs(%out : memref<1x4x4x16xi8>)
  memref.dealloc %q : memref<1x8x8x16xi8>
  memref.dealloc %padded : memref<1x10x10x16xi8>
  return
}

// `pool_padding` is one integer for all four sides, so a border that is not the
// same on both axes is not one the call can express.
// CHECK-LABEL: func.func @asymmetric_border_stays
// CHECK:         linalg.pooling_nhwc_max
func.func @asymmetric_border_stays(%in: memref<1x8x8x16xi8>, %flt: memref<1x1x16x16xi8>,
                                   %w: memref<3x3xi8>, %out: memref<1x4x4x16xi8>) {
  %zi8 = arith.constant 0 : i8
  %min = arith.constant -128 : i8
  %q = memref.alloc() : memref<1x8x8x16xi8>
  gemmlir.conv2d_i8(%in, %flt, %q) {act = #gemmlir.act<relu>, scale = 2.500000e-02 : f32}
    : (memref<1x8x8x16xi8>, memref<1x1x16x16xi8>, memref<1x8x8x16xi8>)
  %padded = memref.alloc() : memref<1x10x10x16xi8>
  linalg.fill ins(%zi8 : i8) outs(%padded : memref<1x10x10x16xi8>)
  %mid = memref.subview %padded[0, 2, 1, 0] [1, 8, 8, 16] [1, 1, 1, 1]
       : memref<1x10x10x16xi8> to memref<1x8x8x16xi8, strided<[1600, 160, 16, 1], offset: 336>>
  memref.copy %q, %mid : memref<1x8x8x16xi8> to memref<1x8x8x16xi8, strided<[1600, 160, 16, 1], offset: 336>>
  linalg.fill ins(%min : i8) outs(%out : memref<1x4x4x16xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%padded, %w : memref<1x10x10x16xi8>, memref<3x3xi8>) outs(%out : memref<1x4x4x16xi8>)
  memref.dealloc %q : memref<1x8x8x16xi8>
  memref.dealloc %padded : memref<1x10x10x16xi8>
  return
}

// A padding the window cannot reach across leaves an output window that is
// entirely padding -- there is no real element for the zero to lose to, so the
// call would invent a value the pool does not have.
// CHECK-LABEL: func.func @window_narrower_than_the_border
// CHECK:         linalg.pooling_nhwc_max
func.func @window_narrower_than_the_border(%in: memref<1x8x8x16xi8>, %flt: memref<1x1x16x16xi8>,
                                           %w: memref<2x2xi8>, %out: memref<1x5x5x16xi8>) {
  %zi8 = arith.constant 0 : i8
  %min = arith.constant -128 : i8
  %q = memref.alloc() : memref<1x8x8x16xi8>
  gemmlir.conv2d_i8(%in, %flt, %q) {act = #gemmlir.act<relu>, scale = 2.500000e-02 : f32}
    : (memref<1x8x8x16xi8>, memref<1x1x16x16xi8>, memref<1x8x8x16xi8>)
  %padded = memref.alloc() : memref<1x12x12x16xi8>
  linalg.fill ins(%zi8 : i8) outs(%padded : memref<1x12x12x16xi8>)
  %mid = memref.subview %padded[0, 2, 2, 0] [1, 8, 8, 16] [1, 1, 1, 1]
       : memref<1x12x12x16xi8> to memref<1x8x8x16xi8, strided<[2304, 192, 16, 1], offset: 416>>
  memref.copy %q, %mid : memref<1x8x8x16xi8> to memref<1x8x8x16xi8, strided<[2304, 192, 16, 1], offset: 416>>
  linalg.fill ins(%min : i8) outs(%out : memref<1x5x5x16xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%padded, %w : memref<1x12x12x16xi8>, memref<2x2xi8>) outs(%out : memref<1x5x5x16xi8>)
  memref.dealloc %q : memref<1x8x8x16xi8>
  memref.dealloc %padded : memref<1x12x12x16xi8>
  return
}
