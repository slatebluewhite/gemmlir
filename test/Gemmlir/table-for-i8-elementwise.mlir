// RUN: gemmlir-opt --table-for-i8-elementwise %s | FileCheck %s

// An elementwise chain below an i8 has 256 answers, and they can all be worked
// out at compile time. EfficientNet's SiLU -- `x * sigmoid(x)`, seven
// operations including a `math.exp` at about 65 cycles -- sits directly under
// the i8 `--quantize-unfoldable-tails` produces.
//
// The anchor is a value *inside* the body: fusion has already put the
// quantization, the dequantization and the activation in one region, so the
// byte never reaches memory.

#id = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: memref.global "private" constant @__gemmlir_table_{{[0-9]+}} : memref<256xf32>
// The entry for the byte 0 is silu(0) = 0, and for -128 it is
// -128*0.01 * sigmoid(-128*0.01), which is about -0.2689 * ... -- what matters
// is that the table is there and the exponential is not.
// CHECK-LABEL: func.func @silu_becomes_a_table
// CHECK:         %[[TBL:.*]] = memref.get_global @__gemmlir_table_{{[0-9]+}}
// CHECK:         linalg.generic
// CHECK:           %[[B:.*]] = arith.trunci
// CHECK:           %[[W:.*]] = arith.extsi %[[B]] : i8 to i32
// CHECK:           %[[S:.*]] = arith.addi %[[W]], %{{.*}} : i32
// CHECK:           %[[I:.*]] = arith.index_cast %[[S]]
// CHECK:           %[[V:.*]] = memref.load %[[TBL]][%[[I]]]
// CHECK:           linalg.yield %[[V]]
// CHECK-NOT:     math.exp
func.func @silu_becomes_a_table(%in: memref<4x8xi32>, %out: memref<4x8xf32>) {
  %one = arith.constant 1.000000e+00 : f32
  %s = arith.constant 1.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in : memref<4x8xi32>) outs(%out : memref<4x8xf32>) {
  ^bb0(%a: i32, %o: f32):
    // the quantization the accelerator will do
    %f = arith.sitofp %a : i32 to f32
    %q = arith.divf %f, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %b = arith.trunci %c1 : i32 to i8
    // and the activation below it, all a function of that byte
    %w = arith.sitofp %b : i8 to f32
    %x = arith.mulf %w, %s : f32
    %n = arith.negf %x : f32
    %e = math.exp %n : f32
    %d = arith.addf %e, %one : f32
    %sig = arith.divf %one, %d : f32
    %y = arith.mulf %sig, %x : f32
    linalg.yield %y : f32
  }
  return
}

// A chain that is only a cast is not worth a table -- a load is not cheaper
// than two arithmetic operations, and the table competes for the same cache.
// CHECK-LABEL: func.func @too_short_to_pay
// CHECK-NOT:     memref.load
// CHECK:         arith.sitofp
func.func @too_short_to_pay(%in: memref<4x8xi8>, %out: memref<4x8xf32>) {
  %s = arith.constant 1.000000e-02 : f32
  linalg.generic {indexing_maps = [#id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in : memref<4x8xi8>) outs(%out : memref<4x8xf32>) {
  ^bb0(%b: i8, %o: f32):
    %w = arith.sitofp %b : i8 to f32
    %x = arith.mulf %w, %s : f32
    linalg.yield %x : f32
  }
  return
}

// A body that reads where it is has a different answer per element, so there
// is no table to build.
// CHECK-LABEL: func.func @position_dependent
// CHECK-NOT:     memref.load {{.*}}__gemmlir_table
// CHECK:         math.exp
func.func @position_dependent(%in: memref<4x8xi8>, %out: memref<4x8xf32>) {
  %one = arith.constant 1.000000e+00 : f32
  %s = arith.constant 1.000000e-02 : f32
  linalg.generic {indexing_maps = [#id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in : memref<4x8xi8>) outs(%out : memref<4x8xf32>) {
  ^bb0(%b: i8, %o: f32):
    %k = linalg.index 1 : index
    %ki = arith.index_cast %k : index to i32
    %kf = arith.sitofp %ki : i32 to f32
    %w = arith.sitofp %b : i8 to f32
    %x = arith.mulf %w, %s : f32
    %n = arith.negf %x : f32
    %e = math.exp %n : f32
    %d = arith.addf %e, %kf : f32
    %y = arith.divf %one, %d : f32
    linalg.yield %y : f32
  }
  return
}

// -----

// **Three** tables in one region.
//
// An LSTM's gates arrive fused: one `linalg.generic` reading three i8 slices of
// the same buffer and computing `sigmoid(f) * c + sigmoid(i) * tanh(g)`. The
// value it yields depends on all three bytes at once, so asking whether *that*
// is a function of one byte answers no -- and the whole gate stayed in software,
// 19.9 ms of the model's 37.6.
//
// Each activation inside it is a function of exactly one byte, though, and each
// of those is 256 answers. The pass takes the **largest** value in the region
// that depends on one byte, whichever value that is, and the driver comes back
// for the next one. **37.6 -> 17.6 ms.**
//
// Two things make the second round possible: a table this pass has already put
// in is a `memref.load` from a constant global, which the purity screen has to
// admit, and the chain it replaced is still in the region until the next
// canonicalization -- evaluable, useless, and tabled again forever if its lack
// of uses is not noticed.

// RUN: gemmlir-opt --table-for-i8-elementwise %s | FileCheck %s --check-prefix=GATES

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// GATES-LABEL: func.func @three_gates_one_region
// Three of them, one per activation.
// GATES:         memref.get_global @__gemmlir_table_
// GATES:         memref.get_global @__gemmlir_table_
// GATES:         memref.get_global @__gemmlir_table_
// GATES-NOT:     math.exp
// GATES-NOT:     math.tanh
func.func @three_gates_one_region(%f: memref<1x48xi8>, %i: memref<1x48xi8>,
                                  %g: memref<1x48xi8>, %out: memref<1x48xf32>) {
  %s = arith.constant 2.500000e-02 : f32
  %one = arith.constant 1.000000e+00 : f32
  %c = arith.constant 3.000000e-01 : f32
  linalg.generic {indexing_maps = [#id2, #id2, #id2, #id2],
                  iterator_types = ["parallel", "parallel"]}
      ins(%f, %i, %g : memref<1x48xi8>, memref<1x48xi8>, memref<1x48xi8>)
      outs(%out : memref<1x48xf32>) {
  ^bb0(%bf: i8, %bi: i8, %bg: i8, %o: f32):
    %wf = arith.sitofp %bf : i8 to f32
    %wi = arith.sitofp %bi : i8 to f32
    %wg = arith.sitofp %bg : i8 to f32
    %df = arith.mulf %wf, %s : f32
    %di = arith.mulf %wi, %s : f32
    %dg = arith.mulf %wg, %s : f32
    %nf = arith.negf %df : f32
    %ef = math.exp %nf : f32
    %af = arith.addf %ef, %one : f32
    %sf = arith.divf %one, %af : f32
    %ni = arith.negf %di : f32
    %ei = math.exp %ni : f32
    %ai = arith.addf %ei, %one : f32
    %si = arith.divf %one, %ai : f32
    %tg = math.tanh %dg : f32
    %l = arith.mulf %sf, %c : f32
    %r = arith.mulf %si, %tg : f32
    %y = arith.addf %l, %r : f32
    linalg.yield %y : f32
  }
  return
}

// -----

// A constant the body reads at a **wider** type. `--fold-batch-norm` leaves the
// epsilon as an f64 `arith.constant` outside the region and an `arith.truncf`
// to f32 inside it; rounding the f64 value once is exactly what that pair
// computes, so the table is still what the loop would have produced.
//
// Evaluating the f64 constant as if it were f32 is not an option --
// `APFloat::convertToFloat` asserts on it -- and DenseNet-121 is the model that
// found it. Note where the crash was: `evaluate` runs on **every** candidate
// value, long before the worth test decides whether to build a table at all, so
// a body carrying a wide constant aborted the pass whether or not it was ever
// going to be tabled.
// CHECK-LABEL: func.func @a_wide_constant_in_the_body
// CHECK:         %[[TBL:.*]] = memref.get_global @__gemmlir_table_{{[0-9]+}}
// CHECK:         linalg.generic
// CHECK:           memref.load %[[TBL]]
// CHECK-NOT:     arith.truncf
// CHECK-NOT:     math.sqrt
func.func @a_wide_constant_in_the_body(%in: memref<4x8xi32>, %out: memref<4x8xf32>) {
  %eps = arith.constant 1.000000e-05 : f64
  %s = arith.constant 1.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in : memref<4x8xi32>) outs(%out : memref<4x8xf32>) {
  ^bb0(%a: i32, %o: f32):
    %f = arith.sitofp %a : i32 to f32
    %q = arith.divf %f, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %b = arith.trunci %c1 : i32 to i8
    // and below the byte, a normalization's `x / sqrt(v + eps)`
    %w = arith.extsi %b : i8 to i32
    %d = arith.sitofp %w : i32 to f32
    %v = arith.mulf %d, %d : f32
    %e = arith.truncf %eps : f64 to f32
    %g = arith.addf %v, %e : f32
    %h = math.sqrt %g : f32
    %k = arith.divf %d, %h : f32
    linalg.yield %k : f32
  }
  return
}
