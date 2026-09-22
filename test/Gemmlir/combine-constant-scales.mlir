// RUN: gemmlir-opt --combine-constant-scales --canonicalize --split-input-file %s | FileCheck %s

// A quantization tail carries its constants one operation at a time:
//
//     fcvt.s.w  fa3, a1              # the accumulator
//     fmadd.s   fa3, fa3, fa5, fs1   # * input scale + bias
//     fmul.s    fa3, fa3, fa4        # / output scale
//     fcvt.w.s  s0, fa3, rne
//
// Both coefficients are known at compile time and both sit on the dependency
// chain, which is what an in-order single-issue core pays for.
// `(a*x + b)*k` is `a*k*x + b*k`.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// 0.025 / 13 == 0.00192307692
// CHECK-LABEL: func.func @a_two_scales_become_one
// CHECK:       %[[K:.*]] = arith.constant 0.00192307692 : f32
// CHECK:         arith.sitofp
// CHECK-NEXT:    arith.mulf %{{.*}}, %[[K]]
// CHECK-NEXT:    math.roundeven
// CHECK-NOT:     arith.divf
func.func @a_two_scales_become_one(%acc: memref<17x576xi32>, %out: memref<17x576xi8>) {
  %c1 = arith.constant 2.500000e-02 : f32
  %c2 = arith.constant 0.000000e+00 : f32
  %c3 = arith.constant 1.300000e+01 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<17x576xi32>) outs(%out : memref<17x576xi8>) {
  ^bb0(%in: i32, %o: i8):
    %f = arith.sitofp %in : i32 to f32
    %m = math.fma %f, %c1, %c2 : f32
    %d = arith.divf %m, %c3 : f32
    %r = math.roundeven %d : f32
    %q = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %q, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// A non-zero offset survives as the addend of one `math.fma`: `(x*2 + 3)*5` is
// `x*10 + 15`.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @b_an_offset_stays_an_offset
// CHECK-DAG:   %[[A:.*]] = arith.constant 1.000000e+01 : f32
// CHECK-DAG:   %[[B:.*]] = arith.constant 1.500000e+01 : f32
// CHECK:       math.fma %{{.*}}, %[[A]], %[[B]]
// CHECK-NOT:   arith.mulf
func.func @b_an_offset_stays_an_offset(%acc: memref<8xf32>, %out: memref<8xi8>) {
  %two = arith.constant 2.000000e+00 : f32
  %three = arith.constant 3.000000e+00 : f32
  %five = arith.constant 5.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%acc : memref<8xf32>) outs(%out : memref<8xi8>) {
  ^bb0(%in: f32, %o: i8):
    %m = arith.mulf %in, %two : f32
    %p = arith.addf %m, %three : f32
    %s = arith.mulf %p, %five : f32
    %r = math.roundeven %s : f32
    %q = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %q, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// A relu in the middle is monotone, so the rounding boundary is all that a few
// ulps can move -- but the chain stops at it, because the relu is not a step of
// `a*x + b`. What is above it and what is below it fold separately.

#id2 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @c_a_relu_splits_the_chain
// CHECK:       arith.mulf
// CHECK:       arith.maxnumf
// CHECK:       arith.mulf
// CHECK-NOT:   arith.mulf
func.func @c_a_relu_splits_the_chain(%acc: memref<8xf32>, %out: memref<8xi8>) {
  %zero = arith.constant 0.000000e+00 : f32
  %two = arith.constant 2.000000e+00 : f32
  %three = arith.constant 3.000000e+00 : f32
  %five = arith.constant 5.000000e+00 : f32
  %seven = arith.constant 7.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel"]}
      ins(%acc : memref<8xf32>) outs(%out : memref<8xi8>) {
  ^bb0(%in: f32, %o: i8):
    %m = arith.mulf %in, %two : f32
    %n = arith.mulf %m, %three : f32
    %z = arith.maxnumf %n, %zero : f32
    %p = arith.mulf %z, %five : f32
    %s = arith.mulf %p, %seven : f32
    %r = math.roundeven %s : f32
    %q = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %q, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// The licence is that the value reaches nothing but a conversion to an integer.
// A chain that ends in a stored `f32` keeps every rounding it had.

#id1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @d_a_float_result_keeps_its_rounding
// CHECK-COUNT-2: arith.mulf
func.func @d_a_float_result_keeps_its_rounding(%in: memref<8xf32>, %out: memref<8xf32>) {
  %two = arith.constant 2.000000e+00 : f32
  %three = arith.constant 3.000000e+00 : f32
  linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
      ins(%in : memref<8xf32>) outs(%out : memref<8xf32>) {
  ^bb0(%a: f32, %o: f32):
    %m = arith.mulf %a, %two : f32
    %n = arith.mulf %m, %three : f32
    linalg.yield %n : f32
  }
  return
}

// -----

// Folding two constants together is the one way a few ulps could become a
// different answer, so an overflow to infinity is refused outright.

#id1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @e_an_overflowing_fold_is_refused
// CHECK-COUNT-2: arith.mulf
func.func @e_an_overflowing_fold_is_refused(%in: memref<8xf32>, %out: memref<8xi8>) {
  %big = arith.constant 3.000000e+38 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
      ins(%in : memref<8xf32>) outs(%out : memref<8xi8>) {
  ^bb0(%a: f32, %o: i8):
    %m = arith.mulf %a, %big : f32
    %n = arith.mulf %m, %big : f32
    %r = math.roundeven %n : f32
    %q = arith.fptosi %r : f32 to i32
    %x = arith.maxsi %q, %lo : i32
    %y = arith.minsi %x, %hi : i32
    %t = arith.trunci %y : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// A step whose other operand is not a constant is not a step: a per-channel
// coefficient is a buffer read, and `--combine-channel-affine` already put the
// constants it could into it.

#id1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @f_a_running_coefficient_is_not_a_constant
// CHECK:       arith.mulf %{{.*}}, %in
// CHECK:       arith.mulf
func.func @f_a_running_coefficient_is_not_a_constant(%in: memref<8xf32>, %k: memref<8xf32>,
                                                     %out: memref<8xi8>) {
  %three = arith.constant 3.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id1, #id1, #id1], iterator_types = ["parallel"]}
      ins(%in, %k : memref<8xf32>, memref<8xf32>) outs(%out : memref<8xi8>) {
  ^bb0(%a: f32, %c: f32, %o: i8):
    %m = arith.mulf %a, %c : f32
    %n = arith.mulf %m, %three : f32
    %r = math.roundeven %n : f32
    %q = arith.fptosi %r : f32 to i32
    %x = arith.maxsi %q, %lo : i32
    %y = arith.minsi %x, %hi : i32
    %t = arith.trunci %y : i32 to i8
    linalg.yield %t : i8
  }
  return
}
