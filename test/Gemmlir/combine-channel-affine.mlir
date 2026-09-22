// RUN: gemmlir-opt --combine-channel-affine --split-input-file %s | FileCheck %s

// A batch norm that could not fold into anybody's weights spends three
// operations an element on numbers that only change per channel:
//
//     fsub.s  fa3, fa3, fa2        # x - mean[c]
//     fmadd.s fa3, fa3, fa1, fa5   # * rsqrt[c] + beta
//     ...relu...
//     fmul.s  fa3, fa3, fa4        # / scale
//     fcvt.w.s a1, fa3, rne
//
// DenseNet-121 is **593 of them** -- every dense layer renormalizes the whole
// concatenated stack and the parameters belong to the *consumer*, so there are
// no weights to fold into. `fsub.s` and `fmul.s` alone are 11.6% of the model
// by program-counter sampling, and they are two of the five steps on the
// dependency chain, which is what an in-order core actually pays.
//
// `((x - m) * r + b) / s` is `x * (r/s) + (b - m*r)/s`.

// CHECK-LABEL: func.func @a_norm_tail_combines
// The per-channel loop writes both coefficients:
// CHECK:         %[[A:.*]] = memref.alloc
// CHECK:         %[[B:.*]] = memref.alloc
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg1, %arg2 : memref<8xf32>, memref<8xf32>)
// CHECK-SAME:      outs(%[[A]], %[[B]] : memref<8xf32>, memref<8xf32>)
// CHECK:           %[[SA:.*]] = arith.divf %{{.*}}, %{{.*}} : f32
// CHECK:           %[[MR:.*]] = arith.mulf
// CHECK:           %[[NUM:.*]] = arith.subf
// CHECK:           %[[SB:.*]] = arith.divf %[[NUM]]
// CHECK:           linalg.yield %[[SA]], %[[SB]]
//
// and the element loop is one multiply-add with the scale gone:
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0, %[[B]], %[[A]]
// CHECK:         ^bb0(%[[X:.*]]: f32, %[[BB:.*]]: f32, %[[AA:.*]]: f32
// CHECK-NEXT:      %[[P:.*]] = arith.mulf %[[X]], %[[AA]]
// CHECK-NEXT:      %[[Y:.*]] = arith.addf %[[P]], %[[BB]]
// CHECK-NEXT:      arith.cmpf ugt, %[[Y]]
// CHECK-NEXT:      arith.select
// CHECK-NEXT:      math.roundeven
// CHECK-NOT:       arith.subf
// CHECK-NOT:       arith.divf
#full = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>
func.func @a_norm_tail_combines(%x: memref<1x4x4x8xf32>, %mean: memref<8xf32>,
                                %rsqrt: memref<8xf32>, %out: memref<1x4x4x8xi8>) {
  %zero = arith.constant 0.000000e+00 : f32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#full, #chan, #chan, #full],
                  iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%x, %mean, %rsqrt : memref<1x4x4x8xf32>, memref<8xf32>, memref<8xf32>)
    outs(%out : memref<1x4x4x8xi8>) {
  ^bb0(%v: f32, %m: f32, %r: f32, %o: i8):
    %c = arith.subf %v, %m : f32
    %n = arith.mulf %c, %r : f32
    %b = arith.addf %n, %zero : f32
    %p = arith.cmpf ugt, %b, %zero : f32
    %relu = arith.select %p, %b, %zero : f32
    %q = arith.divf %relu, %s : f32
    %rd = math.roundeven %q : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// The scale has to be positive for `max(y, 0)/s == max(y/s, 0)`. A negative one
// turns the activation upside down.
// CHECK-LABEL: func.func @a_negative_scale_is_refused
// CHECK:         arith.subf
// CHECK:         arith.mulf
#full = affine_map<(d0, d1) -> (d0, d1)>
#chan = affine_map<(d0, d1) -> (d1)>
func.func @a_negative_scale_is_refused(%x: memref<4x8xf32>, %mean: memref<8xf32>,
                                       %rsqrt: memref<8xf32>, %out: memref<4x8xi8>) {
  %zero = arith.constant 0.000000e+00 : f32
  %s = arith.constant -2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#full, #chan, #chan, #full],
                  iterator_types = ["parallel","parallel"]}
    ins(%x, %mean, %rsqrt : memref<4x8xf32>, memref<8xf32>, memref<8xf32>)
    outs(%out : memref<4x8xi8>) {
  ^bb0(%v: f32, %m: f32, %r: f32, %o: i8):
    %c = arith.subf %v, %m : f32
    %n = arith.mulf %c, %r : f32
    %b = arith.addf %n, %zero : f32
    %p = arith.cmpf ugt, %b, %zero : f32
    %relu = arith.select %p, %b, %zero : f32
    %q = arith.mulf %relu, %s : f32
    %rd = math.roundeven %q : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// The two forms differ by rounding, so the value has to reach a conversion to
// an integer and nothing else -- the same condition and the same argument as
// `--hoist-invariant-reciprocal`. A tail that keeps its f32 answer does not
// qualify.
// CHECK-LABEL: func.func @an_f32_result_is_refused
// CHECK:         arith.subf
// CHECK:         arith.divf
#full = affine_map<(d0, d1) -> (d0, d1)>
#chan = affine_map<(d0, d1) -> (d1)>
func.func @an_f32_result_is_refused(%x: memref<4x8xf32>, %mean: memref<8xf32>,
                                    %rsqrt: memref<8xf32>, %out: memref<4x8xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  %s = arith.constant 2.000000e-02 : f32
  linalg.generic {indexing_maps = [#full, #chan, #chan, #full],
                  iterator_types = ["parallel","parallel"]}
    ins(%x, %mean, %rsqrt : memref<4x8xf32>, memref<8xf32>, memref<8xf32>)
    outs(%out : memref<4x8xf32>) {
  ^bb0(%v: f32, %m: f32, %r: f32, %o: f32):
    %c = arith.subf %v, %m : f32
    %n = arith.mulf %c, %r : f32
    %b = arith.addf %n, %zero : f32
    %p = arith.cmpf ugt, %b, %zero : f32
    %relu = arith.select %p, %b, %zero : f32
    %q = arith.divf %relu, %s : f32
    linalg.yield %q : f32
  }
  return
}

// -----

// The two per-channel operands have to be read the same way, or one loop cannot
// produce both coefficients.
// CHECK-LABEL: func.func @different_maps_are_refused
// CHECK:         arith.subf
#full2 = affine_map<(d0, d1) -> (d0, d1)>
#chan2 = affine_map<(d0, d1) -> (d1)>
#row2 = affine_map<(d0, d1) -> (d0)>
func.func @different_maps_are_refused(%x: memref<8x8xf32>, %mean: memref<8xf32>,
                                      %rsqrt: memref<8xf32>, %out: memref<8x8xi8>) {
  %zero = arith.constant 0.000000e+00 : f32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#full2, #row2, #chan2, #full2],
                  iterator_types = ["parallel","parallel"]}
    ins(%x, %mean, %rsqrt : memref<8x8xf32>, memref<8xf32>, memref<8xf32>)
    outs(%out : memref<8x8xi8>) {
  ^bb0(%v: f32, %m: f32, %r: f32, %o: i8):
    %c = arith.subf %v, %m : f32
    %n = arith.mulf %c, %r : f32
    %b = arith.addf %n, %zero : f32
    %p = arith.cmpf ugt, %b, %zero : f32
    %relu = arith.select %p, %b, %zero : f32
    %q = arith.divf %relu, %s : f32
    %rd = math.roundeven %q : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}
