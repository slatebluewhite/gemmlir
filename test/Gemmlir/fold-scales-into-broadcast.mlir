// RUN: gemmlir-opt --fold-scales-into-broadcast --canonicalize --split-input-file %s | FileCheck %s

// EfficientNet's squeeze-and-excitation is
//
//     (x * input_scale) * gate[c] / output_scale
//
// -- three floating-point multiplies an element, two of them by numbers the
// compiler knows. `gate` is 96 numbers against 16x16x96 elements, so the two
// constants belong in it. It is 26% of that model's elementwise work, and
// `--combine-constant-scales` cannot reach it: the running coefficient sits
// between the two constants and breaks the chain.

#gate = affine_map<(d0, d1, d2, d3) -> (d0, d3, 0, 0)>
#id4  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// 0.02 / 0.05 in f32 is 0.399999976, and the constant is folded once here
// rather than once an element.
// CHECK-LABEL: func.func @a_squeeze_and_excitation
// CHECK:       %[[K:.*]] = arith.constant 0.399999976 : f32
// CHECK:       %[[S:.*]] = memref.alloc() {{.*}} memref<1x96x1x1xf32>
// CHECK:       linalg.generic
// CHECK-SAME:  outs(%[[S]]
// CHECK:         arith.mulf %{{.*}}, %[[K]]
// CHECK:       linalg.generic
// CHECK-SAME:  ins(%[[S]]
// CHECK:         arith.sitofp
// CHECK-NEXT:    arith.mulf
// CHECK-NEXT:    math.roundeven
func.func @a_squeeze_and_excitation(%gate: memref<1x96x1x1xf32>, %x: memref<1x16x16x96xi8>,
                                    %out: memref<1x16x16x96xi8>) {
  %in = arith.constant 2.000000e-02 : f32
  %sc = arith.constant 5.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#gate, #id4, #id4],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%gate, %x : memref<1x96x1x1xf32>, memref<1x16x16x96xi8>)
      outs(%out : memref<1x16x16x96xi8>) {
  ^bb0(%g: f32, %q: i8, %o: i8):
    %f = arith.sitofp %q : i8 to f32
    %m = arith.mulf %f, %in : f32
    %p = arith.mulf %g, %m : f32
    %d = arith.divf %p, %sc : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %i, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// An offset does not commute with the multiply, so it stops the walk: `(x*2+1)*g`
// is not `x*g*2 + ...`.

#gate = affine_map<(d0, d1) -> (d1)>
#id2  = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @b_an_offset_stops_it
// CHECK-NOT:   memref.alloc
// CHECK:       arith.addf
func.func @b_an_offset_stops_it(%gate: memref<8xf32>, %x: memref<64x8xf32>,
                                %out: memref<64x8xi8>) {
  %two = arith.constant 2.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#gate, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%gate, %x : memref<8xf32>, memref<64x8xf32>) outs(%out : memref<64x8xi8>) {
  ^bb0(%g: f32, %v: f32, %o: i8):
    %m = arith.mulf %v, %two : f32
    %p = arith.addf %m, %one : f32
    %q = arith.mulf %g, %p : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %i, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// The point is to scale the operand there are fewer of. An operand the same
// size as the loop buys nothing.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @c_an_operand_the_same_size
// CHECK-NOT:   memref.alloc
// CHECK-COUNT-2: arith.mulf
func.func @c_an_operand_the_same_size(%g: memref<64x8xf32>, %x: memref<64x8xf32>,
                                      %out: memref<64x8xi8>) {
  %two = arith.constant 2.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%g, %x : memref<64x8xf32>, memref<64x8xf32>) outs(%out : memref<64x8xi8>) {
  ^bb0(%a: f32, %v: f32, %o: i8):
    %m = arith.mulf %v, %two : f32
    %q = arith.mulf %a, %m : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %x1 = arith.maxsi %i, %lo : i32
    %y = arith.minsi %x1, %hi : i32
    %t = arith.trunci %y : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// The licence is that the value reaches nothing but a conversion to an integer.
// A stored `f32` keeps every rounding it had.

#gate = affine_map<(d0, d1) -> (d1)>
#id2  = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @d_a_float_result_keeps_its_rounding
// CHECK-NOT:   memref.alloc
// CHECK-COUNT-2: arith.mulf
func.func @d_a_float_result_keeps_its_rounding(%gate: memref<8xf32>, %x: memref<64x8xf32>,
                                               %out: memref<64x8xf32>) {
  %two = arith.constant 2.000000e+00 : f32
  linalg.generic {indexing_maps = [#gate, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%gate, %x : memref<8xf32>, memref<64x8xf32>) outs(%out : memref<64x8xf32>) {
  ^bb0(%g: f32, %v: f32, %o: f32):
    %m = arith.mulf %v, %two : f32
    %q = arith.mulf %g, %m : f32
    linalg.yield %q : f32
  }
  return
}

// -----

// A layer norm is `(x - mean) * rstd + beta` over an output scale, and once
// `--sink-elementwise-into-readers` puts the pieces in one loop that is four
// floating-point operations an element for one multiply's worth of work. The
// offset is a number the compiler has, so it scales with everything after it:
// `(x*r + b)/s` is `x*(r/s) + b/s`.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @layer_norm_offset_moves
// beta / output_scale is 3/4, and the reciprocal is scaled by 1/4:
// CHECK-DAG:   %[[B:.*]] = arith.constant 7.500000e-01 : f32
// CHECK-DAG:   %[[K:.*]] = arith.constant 2.500000e-01 : f32
// CHECK:       %[[S:.*]] = memref.alloc() {{.*}} memref<8xf32>
// CHECK:       linalg.generic {{.*}} ins(%arg2 {{.*}} outs(%[[S]]
// CHECK:         arith.mulf %{{.*}}, %[[K]]
// The element loop keeps a subtract, a multiply and the folded offset -- three
// operations where there were four, and `--fuse-multiply-add` makes the last
// two one `math.fma`.
// CHECK:       linalg.generic {{.*}} ins(%arg0, %arg1, %[[S]]
// CHECK:         arith.subf
// CHECK-NEXT:    arith.mulf
// CHECK-NEXT:    arith.addf %{{.*}}, %[[B]]
// CHECK-NOT:     arith.divf
// CHECK:         math.roundeven
func.func @layer_norm_offset_moves(%x: memref<8x64xf32>, %mean: memref<8xf32>,
                                   %rstd: memref<8xf32>, %out: memref<8x64xi8>) {
  %beta = arith.constant 3.000000e+00 : f32
  %oscale = arith.constant 4.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id2, #row, #row, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x, %mean, %rstd : memref<8x64xf32>, memref<8xf32>, memref<8xf32>)
      outs(%out : memref<8x64xi8>) {
  ^bb0(%in: f32, %m: f32, %r: f32, %o: i8):
    %c = arith.subf %in, %m : f32
    %n = arith.mulf %c, %r : f32
    %b = arith.addf %n, %beta : f32
    %q = arith.divf %b, %oscale : f32
    %e = math.roundeven %q : f32
    %i = arith.fptosi %e : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// An offset *below* the multiply stays where it is -- there is no constant to
// put in the buffer for it -- but it does not stop the scale above from
// moving, because `((x + c) * r) / s` is `(x + c) * (r/s)`.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @offset_below_stays
// CHECK:       linalg.generic {{.*}} ins(%arg1
// CHECK:         arith.mulf
// CHECK:       linalg.generic {{.*}} ins(%arg0
// CHECK:         arith.addf
// CHECK-NEXT:    arith.mulf
// CHECK-NOT:     arith.divf
// CHECK:         math.roundeven
func.func @offset_below_stays(%x: memref<8x64xf32>, %rstd: memref<8xf32>,
                                %out: memref<8x64xi8>) {
  %shift = arith.constant 3.000000e+00 : f32
  %oscale = arith.constant 4.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id2, #row, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x, %rstd : memref<8x64xf32>, memref<8xf32>)
      outs(%out : memref<8x64xi8>) {
  ^bb0(%in: f32, %r: f32, %o: i8):
    %s = arith.addf %in, %shift : f32
    %n = arith.mulf %s, %r : f32
    %q = arith.divf %n, %oscale : f32
    %e = math.roundeven %q : f32
    %i = arith.fptosi %e : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}
