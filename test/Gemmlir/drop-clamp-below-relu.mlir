// RUN: gemmlir-opt --drop-clamp-below-relu --split-input-file %s | FileCheck %s

// A quantization tail is `clamp(round(x / s), -128, 127)`, and on a core with no
// `Zbb` each end of that clamp is a compare, a branch and a `li`. Where the
// value came through a relu it is already at least zero, so the lower end can
// never bite: `arith.maxnumf(x, 0)` hands back the operand that is **not** a
// NaN, so its result is non-negative whatever `x` was, and neither
// `math.roundeven` nor a multiply by a positive scale can take it below zero.
//
// DenseNet-121's `forward` goes from 118,091 instructions to 112,718.
//
// This is not the rewrite measured and refused twice -- that one moved the clamp
// into the float, which puts two more operations *on* the dependency chain.
// This takes two instructions away and adds nothing.

// CHECK-LABEL: func.func @below_a_relu
// CHECK:         arith.fptosi
// CHECK-NOT:     arith.maxsi
// CHECK:         arith.minsi
func.func @below_a_relu(%x: f32) -> i8 {
  %z = arith.constant 0.000000e+00 : f32
  %s = arith.constant 5.000000e+01 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %r = arith.maxnumf %x, %z : f32
  %m = arith.mulf %r, %s : f32
  %e = math.roundeven %m : f32
  %i = arith.fptosi %e : f32 to i32
  %a = arith.maxsi %i, %lo : i32
  %b = arith.minsi %a, %hi : i32
  %t = arith.trunci %b : i32 to i8
  return %t : i8
}

// -----

// Without the activation the value can be negative and the clamp is real.
// CHECK-LABEL: func.func @without_a_relu
// CHECK:         arith.maxsi
// CHECK:         arith.minsi
func.func @without_a_relu(%x: f32) -> i8 {
  %s = arith.constant 5.000000e+01 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %m = arith.mulf %x, %s : f32
  %e = math.roundeven %m : f32
  %i = arith.fptosi %e : f32 to i32
  %a = arith.maxsi %i, %lo : i32
  %b = arith.minsi %a, %hi : i32
  %t = arith.trunci %b : i32 to i8
  return %t : i8
}

// -----

// A negative scale turns the activation's output upside down, so the lower end
// bites again.
// CHECK-LABEL: func.func @a_negative_scale_keeps_it
// CHECK:         arith.maxsi
func.func @a_negative_scale_keeps_it(%x: f32) -> i8 {
  %z = arith.constant 0.000000e+00 : f32
  %s = arith.constant -5.000000e+01 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %r = arith.maxnumf %x, %z : f32
  %m = arith.mulf %r, %s : f32
  %e = math.roundeven %m : f32
  %i = arith.fptosi %e : f32 to i32
  %a = arith.maxsi %i, %lo : i32
  %b = arith.minsi %a, %hi : i32
  %t = arith.trunci %b : i32 to i8
  return %t : i8
}

// -----

// A bound above zero is a real clamp whatever is above it -- a `leaky` lower
// limit, not the type's minimum.
// CHECK-LABEL: func.func @a_positive_bound_stays
// CHECK:         arith.maxsi
func.func @a_positive_bound_stays(%x: f32) -> i32 {
  %z = arith.constant 0.000000e+00 : f32
  %five = arith.constant 5 : i32
  %r = arith.maxnumf %x, %z : f32
  %i = arith.fptosi %r : f32 to i32
  %a = arith.maxsi %i, %five : i32
  return %a : i32
}

// -----

// `arith.maximumf` propagates a NaN where `maxnumf` does not, so its result is
// not provably non-negative and the clamp stays.
// CHECK-LABEL: func.func @a_propagating_maximum_keeps_it
// CHECK:         arith.maxsi
func.func @a_propagating_maximum_keeps_it(%x: f32) -> i8 {
  %z = arith.constant 0.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %r = arith.maximumf %x, %z : f32
  %i = arith.fptosi %r : f32 to i32
  %a = arith.maxsi %i, %lo : i32
  %b = arith.minsi %a, %hi : i32
  %t = arith.trunci %b : i32 to i8
  return %t : i8
}
