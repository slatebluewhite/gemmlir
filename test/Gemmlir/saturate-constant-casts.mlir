// RUN: gemmlir-opt --saturate-constant-casts --split-input-file %s | FileCheck %s

// **A max-pool's padding is minus infinity, and quantizing it is undefined.**
//
// A padded max-pool arrives from the frontend with `-inf` in its border. When
// the pool moves onto i8 the quantization of that constant is left in front of
// it, and `arith.fptosi` of a value outside the destination's range is
// **poison** -- `maxsi(poison, -128)` is poison too, so the clamp does not
// rescue it. MLIR's folder leaves the operation alone for exactly that reason.
//
// What the border then holds is whatever the backend materializes, and that
// **changes with the surrounding code**: GoogLeNet's answer moved by 0.6% of its
// output range when `--unroll-elementwise-loops` went from two to four, and this
// one byte was why. Neither answer was right; the value was never defined.
//
// Replacing poison with a defined value is a refinement, so folding is always
// legal. Folding it the way the hardware does -- saturate to the extremes -- is
// what makes the compiled answer and the obvious reading of the source agree:
// `INT_MIN`, which the clamp below turns into the i8 minimum, which is the
// pool's own identity.

// CHECK-LABEL: func.func @minus_infinity_saturates
// `INT_MIN` through the clamp is the i8 minimum, and the driver folds the rest
// of the chain away with it -- the border ends up holding the pool's own
// identity, which is what the padding always meant.
// CHECK:         arith.constant -128 : i8
// CHECK-NOT:     arith.fptosi
func.func @minus_infinity_saturates() -> i8 {
  %ninf = arith.constant 0xFF800000 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %i = arith.fptosi %ninf : f32 to i32
  %a = arith.maxsi %i, %lo : i32
  %b = arith.minsi %a, %hi : i32
  %t = arith.trunci %b : i32 to i8
  return %t : i8
}

// -----

// The other end, and the ordinary case in between.
// CHECK-LABEL: func.func @the_other_extremes
// CHECK-DAG:     arith.constant 2147483647 : i32
// CHECK-DAG:     arith.constant 3 : i32
// CHECK-DAG:     arith.constant -5 : i32
// CHECK-NOT:     arith.fptosi
func.func @the_other_extremes() -> (i32, i32, i32) {
  %pinf = arith.constant 0x7F800000 : f32
  %three = arith.constant 3.700000e+00 : f32
  %neg = arith.constant -5.200000e+00 : f32
  %a = arith.fptosi %pinf : f32 to i32
  %b = arith.fptosi %three : f32 to i32
  %c = arith.fptosi %neg : f32 to i32
  return %a, %b, %c : i32, i32, i32
}

// -----

// A NaN is poison too, so any answer is a refinement; the maximum is what
// RISC-V's `fcvt.w.s` hands back, and matching the hardware is the least
// surprising choice.
// CHECK-LABEL: func.func @a_nan_takes_the_maximum
// CHECK:         arith.constant 127 : i8
// CHECK-NOT:     arith.fptosi
func.func @a_nan_takes_the_maximum() -> i8 {
  %nan = arith.constant 0x7FC00000 : f32
  %i = arith.fptosi %nan : f32 to i8
  return %i : i8
}

// -----

// A value that is not a constant is left where it is: this is a fold, not a
// clamp, and inserting a runtime saturation is the rewrite
// `--select-to-minmax` already measured and refused.
// CHECK-LABEL: func.func @a_runtime_value_is_left_alone
// CHECK:         arith.fptosi
func.func @a_runtime_value_is_left_alone(%x: f32) -> i32 {
  %i = arith.fptosi %x : f32 to i32
  return %i : i32
}
