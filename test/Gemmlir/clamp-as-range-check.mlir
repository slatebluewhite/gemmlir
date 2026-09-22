// RUN: gemmlir-opt --clamp-as-range-check --split-input-file %s | FileCheck %s

// Every quantization tail ends `maxsi(v, lo)` then `minsi(v, hi)`. This board is
// plain rv64gc -- no Zbb, so no `max`/`min` and no conditional move -- and the
// pair costs five instructions:
//
//     sgtz a5, a4 / neg a5, a5 / and a5, a5, a4 / li a4, 127 / blt a5, a4, .+
//
// But the pair is a range check, and one *unsigned* compare decides it:
// anything outside [lo, hi] has `(unsigned)(v - lo) > hi - lo`, negatives
// included. `llc` turns the `scf.if` below into one `bltu` with the fixup out
// of line. Measured as a kernel at DenseNet's shape: -5.1%.

// CHECK-LABEL: func.func @a_zero_to_127
// CHECK-DAG:   %[[B:.*]] = arith.constant 128 : i32
// CHECK:       %[[IN:.*]] = arith.cmpi ult, %arg0, %[[B]]
// CHECK:       scf.if %[[IN]]
// CHECK:         scf.yield %arg0
// CHECK:       } else {
// CHECK:         arith.cmpi slt, %arg0
// CHECK:         arith.select
// CHECK-NOT:   arith.maxsi
// CHECK-NOT:   arith.minsi
func.func @a_zero_to_127(%v: i32) -> i32 {
  %lo = arith.constant 0 : i32
  %hi = arith.constant 127 : i32
  %a = arith.maxsi %v, %lo : i32
  %b = arith.minsi %a, %hi : i32
  return %b : i32
}

// -----

// A signed byte's range needs the shift, which is one `addi` and still cheaper
// than the four it replaces.

// CHECK-LABEL: func.func @b_minus_128_to_127
// CHECK-DAG:   %[[LO:.*]] = arith.constant -128 : i32
// CHECK-DAG:   %[[B:.*]] = arith.constant 256 : i32
// CHECK:       %[[S:.*]] = arith.subi %arg0, %[[LO]]
// CHECK:       arith.cmpi ult, %[[S]], %[[B]]
// CHECK:       scf.if
func.func @b_minus_128_to_127(%v: i32) -> i32 {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %a = arith.maxsi %v, %lo : i32
  %b = arith.minsi %a, %hi : i32
  return %b : i32
}

// -----

// A clamp with only one side is already a compare and a branch; there is
// nothing to take away.

// CHECK-LABEL: func.func @c_one_sided
// CHECK:       arith.minsi
// CHECK-NOT:   scf.if
func.func @c_one_sided(%v: i32) -> i32 {
  %hi = arith.constant 127 : i32
  %b = arith.minsi %v, %hi : i32
  return %b : i32
}

// -----

// Bounds that are not constants name no range.

// CHECK-LABEL: func.func @d_running_bounds
// CHECK:       arith.maxsi
// CHECK:       arith.minsi
// CHECK-NOT:   scf.if
func.func @d_running_bounds(%v: i32, %lo: i32, %hi: i32) -> i32 {
  %a = arith.maxsi %v, %lo : i32
  %b = arith.minsi %a, %hi : i32
  return %b : i32
}

// -----

// The shape a quantization tail actually has: the range check sits between the
// conversion and the truncation, and the byte store is what follows.

// CHECK-LABEL: func.func @e_a_whole_tail
// CHECK:       arith.fptosi
// CHECK:       arith.subi
// CHECK:       arith.cmpi ult
// CHECK:       %[[R:.*]] = scf.if
// CHECK:       arith.trunci %[[R]] : i32 to i8
func.func @e_a_whole_tail(%x: f32) -> i8 {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %r = math.roundeven %x : f32
  %i = arith.fptosi %r : f32 to i32
  %a = arith.maxsi %i, %lo : i32
  %b = arith.minsi %a, %hi : i32
  %t = arith.trunci %b : i32 to i8
  return %t : i8
}
