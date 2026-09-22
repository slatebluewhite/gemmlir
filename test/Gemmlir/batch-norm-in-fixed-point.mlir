// RUN: gemmlir-opt --batch-norm-in-fixed-point --split-input-file %s | FileCheck %s

// DenseNet's batch norm is 72% of the model and about 27 cycles an element for
// an eight-instruction body. Shortening it was measured five ways and every one
// bought 1-2%; what it is paying for is the floating point.
//
// The coefficients are compile-time constants and the value they multiply is a
// dequantized byte, so the whole tail is a function of one byte and one
// channel. Measured as a kernel on the board: -16.6%.

#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>

memref.global "private" constant @a_scale : memref<4xf32> = dense<[0.53, 1.47, 0.29, 1.91]>
memref.global "private" constant @a_bias : memref<4xf32> = dense<[1.03, -2.07, 0.41, 0.13]>

// CHECK-LABEL: func.func @a_a_dequantized_batch_norm
// The dequantize loop has nobody left to read it:
// CHECK-NOT:   arith.sitofp
// CHECK-NOT:   math.roundeven
// CHECK:       linalg.generic
// CHECK:         arith.extsi
// CHECK:         arith.muli
// CHECK:         arith.addi
// CHECK:         arith.shrsi
// The relu stays -- the values do go negative -- but the upper clamp does not:
// the same enumeration that proves the two forms agree also says `minsi` never
// fires, because 127/32 * 1.91 + 0.13 is 7.7 and the largest coefficient is the
// one that decides it.
// CHECK:         arith.maxsi
// CHECK-NOT:     arith.minsi
// CHECK:         arith.trunci
func.func @a_a_dequantized_batch_norm(%q: memref<1x2x2x4xi8>, %out: memref<1x2x2x4xi8>) {
  %s = arith.constant 3.125000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %f = memref.alloc() : memref<1x2x2x4xf32>
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q : memref<1x2x2x4xi8>) outs(%f : memref<1x2x2x4xf32>) {
  ^bb0(%in: i8, %o: f32):
    %c = arith.sitofp %in : i8 to f32
    %v = arith.mulf %c, %s : f32
    linalg.yield %v : f32
  }
  %A = memref.get_global @a_scale : memref<4xf32>
  %B = memref.get_global @a_bias : memref<4xf32>
  linalg.generic {indexing_maps = [#nhwc, #chan, #chan, #nhwc],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%f, %B, %A : memref<1x2x2x4xf32>, memref<4xf32>, memref<4xf32>)
      outs(%out : memref<1x2x2x4xi8>) {
  ^bb0(%x: f32, %b: f32, %a: f32, %o: i8):
    %m = arith.mulf %x, %a : f32
    %p = arith.addf %m, %b : f32
    %g = arith.cmpf ugt, %p, %zero : f32
    %r = arith.select %g, %p, %zero : f32
    %e = math.roundeven %r : f32
    %i = arith.fptosi %e : f32 to i32
    %c1 = arith.maxsi %i, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %f : memref<1x2x2x4xf32>
  return
}

// -----

// A relu written as `cmpf` + `select` reads its input **twice**, so `hasOneUse`
// on the value entering it is the wrong test -- it refused all 593 of
// DenseNet's the first time this shape came up. Without the relu the value has
// one use and the walk is the ordinary one.

#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>

memref.global "private" constant @b_scale : memref<4xf32> = dense<[0.53, 1.47, 0.29, 1.91]>
memref.global "private" constant @b_bias : memref<4xf32> = dense<[1.03, -2.07, 0.41, 0.13]>

// CHECK-LABEL: func.func @b_no_relu
// CHECK-NOT:   math.roundeven
// CHECK:         arith.shrsi
func.func @b_no_relu(%q: memref<1x2x2x4xi8>, %out: memref<1x2x2x4xi8>) {
  %s = arith.constant 3.125000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %f = memref.alloc() : memref<1x2x2x4xf32>
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q : memref<1x2x2x4xi8>) outs(%f : memref<1x2x2x4xf32>) {
  ^bb0(%in: i8, %o: f32):
    %c = arith.sitofp %in : i8 to f32
    %v = arith.mulf %c, %s : f32
    linalg.yield %v : f32
  }
  %A = memref.get_global @b_scale : memref<4xf32>
  %B = memref.get_global @b_bias : memref<4xf32>
  linalg.generic {indexing_maps = [#nhwc, #chan, #chan, #nhwc],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%f, %B, %A : memref<1x2x2x4xf32>, memref<4xf32>, memref<4xf32>)
      outs(%out : memref<1x2x2x4xi8>) {
  ^bb0(%x: f32, %b: f32, %a: f32, %o: i8):
    %m = arith.mulf %x, %a : f32
    %p = arith.addf %m, %b : f32
    %e = math.roundeven %p : f32
    %i = arith.fptosi %e : f32 to i32
    %c1 = arith.maxsi %i, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %f : memref<1x2x2x4xf32>
  return
}

// -----

// The rewrite is exact or it does not happen, and there are two ways for it not
// to be. A coefficient large enough that no shift is both precise and safe from
// overflow is one. The other is a **tie**: the shift rounds half away from zero
// and `roundeven` rounds half to even, so a value landing exactly on `n + 1/2`
// disagrees -- 15 of DenseNet's 597 sites are refused for that, and the other
// 582 are rewritten.

#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>

memref.global "private" constant @c_scale : memref<4xf32> = dense<[1.0e+20, 1.0, 1.0, 1.0]>
memref.global "private" constant @c_bias : memref<4xf32> = dense<0.0>

// CHECK-LABEL: func.func @c_no_shift_is_exact
// CHECK:       math.roundeven
func.func @c_no_shift_is_exact(%q: memref<1x2x2x4xi8>, %out: memref<1x2x2x4xi8>) {
  %s = arith.constant 3.125000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %f = memref.alloc() : memref<1x2x2x4xf32>
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q : memref<1x2x2x4xi8>) outs(%f : memref<1x2x2x4xf32>) {
  ^bb0(%in: i8, %o: f32):
    %c = arith.sitofp %in : i8 to f32
    %v = arith.mulf %c, %s : f32
    linalg.yield %v : f32
  }
  %A = memref.get_global @c_scale : memref<4xf32>
  %B = memref.get_global @c_bias : memref<4xf32>
  linalg.generic {indexing_maps = [#nhwc, #chan, #chan, #nhwc],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%f, %B, %A : memref<1x2x2x4xf32>, memref<4xf32>, memref<4xf32>)
      outs(%out : memref<1x2x2x4xi8>) {
  ^bb0(%x: f32, %b: f32, %a: f32, %o: i8):
    %m = arith.mulf %x, %a : f32
    %p = arith.addf %m, %b : f32
    %e = math.roundeven %p : f32
    %i = arith.fptosi %e : f32 to i32
    %c1 = arith.maxsi %i, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %f : memref<1x2x2x4xf32>
  return
}

// -----

// A coefficient that is not a constant is not known at compile time, and then
// nothing can be verified.

#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: func.func @d_a_running_coefficient
// CHECK:       math.roundeven
func.func @d_a_running_coefficient(%q: memref<1x2x2x4xi8>, %A: memref<4xf32>,
                                   %B: memref<4xf32>, %out: memref<1x2x2x4xi8>) {
  %s = arith.constant 3.125000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %f = memref.alloc() : memref<1x2x2x4xf32>
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q : memref<1x2x2x4xi8>) outs(%f : memref<1x2x2x4xf32>) {
  ^bb0(%in: i8, %o: f32):
    %c = arith.sitofp %in : i8 to f32
    %v = arith.mulf %c, %s : f32
    linalg.yield %v : f32
  }
  linalg.generic {indexing_maps = [#nhwc, #chan, #chan, #nhwc],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%f, %B, %A : memref<1x2x2x4xf32>, memref<4xf32>, memref<4xf32>)
      outs(%out : memref<1x2x2x4xi8>) {
  ^bb0(%x: f32, %b: f32, %a: f32, %o: i8):
    %m = arith.mulf %x, %a : f32
    %p = arith.addf %m, %b : f32
    %e = math.roundeven %p : f32
    %i = arith.fptosi %e : f32 to i32
    %c1 = arith.maxsi %i, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %f : memref<1x2x2x4xf32>
  return
}

// -----

// A buffer that is not a dequantized byte -- a pooled f32, say -- has no byte to
// index the answer by.

#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>

memref.global "private" constant @e_scale : memref<4xf32> = dense<[0.53, 1.47, 0.29, 1.91]>
memref.global "private" constant @e_bias : memref<4xf32> = dense<0.0>

// CHECK-LABEL: func.func @e_not_a_dequantized_byte
// CHECK:       math.roundeven
func.func @e_not_a_dequantized_byte(%x: memref<1x2x2x4xf32>, %out: memref<1x2x2x4xi8>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %A = memref.get_global @e_scale : memref<4xf32>
  %B = memref.get_global @e_bias : memref<4xf32>
  linalg.generic {indexing_maps = [#nhwc, #chan, #chan, #nhwc],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x, %B, %A : memref<1x2x2x4xf32>, memref<4xf32>, memref<4xf32>)
      outs(%out : memref<1x2x2x4xi8>) {
  ^bb0(%v: f32, %b: f32, %a: f32, %o: i8):
    %m = arith.mulf %v, %a : f32
    %p = arith.addf %m, %b : f32
    %e = math.roundeven %p : f32
    %i = arith.fptosi %e : f32 to i32
    %c1 = arith.maxsi %i, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// A coefficient big enough to leave the byte's range keeps its clamp: 127/4 *
// 21.0 is 666, so `minsi` is live and dropping it would write a different byte.

#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>

memref.global "private" constant @z_scale : memref<2xf32> = dense<[21.0, 19.0]>
memref.global "private" constant @z_bias : memref<2xf32> = dense<[1.03, -2.07]>

// CHECK-LABEL: func.func @z_clamp_is_live
// CHECK:         arith.shrsi
// CHECK:         arith.maxsi
// CHECK:         arith.minsi
// CHECK:         arith.trunci
func.func @z_clamp_is_live(%q: memref<1x2x2x2xi8>, %out: memref<1x2x2x2xi8>) {
  %s = arith.constant 2.500000e-01 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %f = memref.alloc() : memref<1x2x2x2xf32>
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q : memref<1x2x2x2xi8>) outs(%f : memref<1x2x2x2xf32>) {
  ^bb0(%in: i8, %o: f32):
    %c = arith.sitofp %in : i8 to f32
    %v = arith.mulf %c, %s : f32
    linalg.yield %v : f32
  }
  %A = memref.get_global @z_scale : memref<2xf32>
  %B = memref.get_global @z_bias : memref<2xf32>
  linalg.generic {indexing_maps = [#nhwc, #chan, #chan, #nhwc],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%f, %B, %A : memref<1x2x2x2xf32>, memref<2xf32>, memref<2xf32>)
      outs(%out : memref<1x2x2x2xi8>) {
  ^bb0(%x: f32, %b: f32, %a: f32, %o: i8):
    %m = arith.mulf %x, %a : f32
    %p = arith.addf %m, %b : f32
    %e = math.roundeven %p : f32
    %i = arith.fptosi %e : f32 to i32
    %c1 = arith.maxsi %i, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %f : memref<1x2x2x2xf32>
  return
}
