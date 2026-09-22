// RUN: gemmlir-opt --select-to-minmax %s | FileCheck %s

// An ordered comparison already agrees with `maxnumf` on a NaN -- both hand
// back the other operand -- so nothing has to be proved.

// CHECK-LABEL: func @ordered_relu
// CHECK:         arith.maxnumf %arg0, %{{.*}} : f32
// CHECK-NOT:     arith.cmpf
func.func @ordered_relu(%x: f32) -> f32 {
  %z = arith.constant 0.0 : f32
  %c = arith.cmpf ogt, %x, %z : f32
  %r = arith.select %c, %x, %z : f32
  return %r : f32
}

// The unordered one is what torch-mlir emits, and it keeps a NaN where
// `fmax.s` would return zero. An opaque value is left alone.

// CHECK-LABEL: func @unordered_relu_unproven
// CHECK:         arith.cmpf ugt
// CHECK:         arith.select
// CHECK-NOT:     arith.maxnumf
func.func @unordered_relu_unproven(%x: f32) -> f32 {
  %z = arith.constant 0.0 : f32
  %c = arith.cmpf ugt, %x, %z : f32
  %r = arith.select %c, %x, %z : f32
  return %r : f32
}

// A dequantize tail: an i32 accumulator through `sitofp` is finite, a finite
// scale cannot make a NaN of it, and the bias is a constant global whose
// contents the pass reads. So the unordered form folds here.

#acc = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>
memref.global "private" constant @bias : memref<4xf32> = dense<[1.0, 2.0, 3.0, 4.0]>

// CHECK-LABEL: func @unordered_relu_proven
// CHECK:         linalg.generic
// CHECK:           arith.maxnumf
// CHECK-NOT:       arith.cmpf
func.func @unordered_relu_proven(%acc: memref<8x4xi32>, %out: memref<8x4xf32>) {
  %s = arith.constant 0.013 : f32
  %z = arith.constant 0.0 : f32
  %b = memref.get_global @bias : memref<4xf32>
  linalg.generic {indexing_maps = [#acc, #bias, #acc],
                  iterator_types = ["parallel", "parallel"]}
      ins(%acc, %b: memref<8x4xi32>, memref<4xf32>) outs(%out: memref<8x4xf32>) {
  ^bb0(%a: i32, %bb: f32, %o: f32):
    %f = arith.sitofp %a : i32 to f32
    %p = math.fma %f, %s, %bb : f32
    %c = arith.cmpf ugt, %p, %z : f32
    %r = arith.select %c, %p, %z : f32
    linalg.yield %r : f32
  }
  return
}

// The same tail with a NaN sitting in the bias: there is nothing to prove and
// the select stays.

memref.global "private" constant @nanbias : memref<4xf32> = dense<[1.0, 0x7FC00000, 3.0, 4.0]>

// CHECK-LABEL: func @bias_holds_a_nan
// CHECK:         linalg.generic
// CHECK:           arith.cmpf ugt
// CHECK:           arith.select
// CHECK-NOT:       arith.maxnumf
func.func @bias_holds_a_nan(%acc: memref<8x4xi32>, %out: memref<8x4xf32>) {
  %s = arith.constant 0.013 : f32
  %z = arith.constant 0.0 : f32
  %b = memref.get_global @nanbias : memref<4xf32>
  linalg.generic {indexing_maps = [#acc, #bias, #acc],
                  iterator_types = ["parallel", "parallel"]}
      ins(%acc, %b: memref<8x4xi32>, memref<4xf32>) outs(%out: memref<8x4xf32>) {
  ^bb0(%a: i32, %bb: f32, %o: f32):
    %f = arith.sitofp %a : i32 to f32
    %p = math.fma %f, %s, %bb : f32
    %c = arith.cmpf ugt, %p, %z : f32
    %r = arith.select %c, %p, %z : f32
    linalg.yield %r : f32
  }
  return
}

// A clamp from above is the minimum, and an ordered one needs no proof.

// CHECK-LABEL: func @ordered_min
// CHECK:         arith.minnumf %arg0, %{{.*}} : f32
func.func @ordered_min(%x: f32) -> f32 {
  %six = arith.constant 6.0 : f32
  %c = arith.cmpf olt, %x, %six : f32
  %r = arith.select %c, %x, %six : f32
  return %r : f32
}

// With the select's arms the other way round, a greater-than is the minimum.
// Unordered, so what has to not be a NaN is the true value -- the constant.

// CHECK-LABEL: func @swapped
// CHECK:         arith.minnumf %arg0, %{{.*}} : f32
// CHECK-NOT:     arith.select
func.func @swapped(%x: f32) -> f32 {
  %six = arith.constant 6.0 : f32
  %c = arith.cmpf ugt, %x, %six : f32
  %r = arith.select %c, %six, %x : f32
  return %r : f32
}

// Written the other way round with an *ordered* predicate, the value the
// select falls to on a NaN is the variable, so this one does need it proved
// and an opaque argument is left alone.

// CHECK-LABEL: func @swapped_ordered
// CHECK:         arith.cmpf ogt
// CHECK:         arith.select
// CHECK-NOT:     arith.minnumf
func.func @swapped_ordered(%x: f32) -> f32 {
  %six = arith.constant 6.0 : f32
  %c = arith.cmpf ogt, %x, %six : f32
  %r = arith.select %c, %six, %x : f32
  return %r : f32
}

// `fmax.s(+0, -0)` is `+0`, where the select keeps the `-0` it compared
// against. A maximum against a negative zero, and a minimum against a positive
// one, are left alone.

// CHECK-LABEL: func @max_against_negative_zero
// CHECK:         arith.select
// CHECK-NOT:     arith.maxnumf
func.func @max_against_negative_zero(%x: f32) -> f32 {
  %nz = arith.constant -0.0 : f32
  %c = arith.cmpf ogt, %x, %nz : f32
  %r = arith.select %c, %x, %nz : f32
  return %r : f32
}

// CHECK-LABEL: func @min_against_positive_zero
// CHECK:         arith.select
// CHECK-NOT:     arith.minnumf
func.func @min_against_positive_zero(%x: f32) -> f32 {
  %z = arith.constant 0.0 : f32
  %c = arith.cmpf olt, %x, %z : f32
  %r = arith.select %c, %x, %z : f32
  return %r : f32
}

// A comparison something else reads has to stay, so folding the select would
// not remove it.

// CHECK-LABEL: func @compare_read_twice
// CHECK:         arith.cmpf
// CHECK:         arith.select
// CHECK-NOT:     arith.maxnumf
func.func @compare_read_twice(%x: f32) -> (f32, i1) {
  %z = arith.constant 0.0 : f32
  %c = arith.cmpf ogt, %x, %z : f32
  %r = arith.select %c, %x, %z : f32
  return %r, %c : f32, i1
}

// A select whose arms are not the two values compared is not a min or a max.

// CHECK-LABEL: func @unrelated_arms
// CHECK:         arith.select
// CHECK-NOT:     arith.maxnumf
func.func @unrelated_arms(%x: f32, %y: f32) -> f32 {
  %z = arith.constant 0.0 : f32
  %c = arith.cmpf ogt, %x, %z : f32
  %r = arith.select %c, %y, %z : f32
  return %r : f32
}

// -----

// **Tried and reverted.** A relu whose input nothing proves finite, but whose
// result reaches nothing but a conversion to an integer, could be taken: the
// two forms differ only on a NaN and `fptosi` of a NaN is poison already.
// It lets a batch norm reading an f32 buffer end in one `fmax.s` -- all 598 of
// DenseNet's compare-and-selects over 8.0 million elements.
//
// Correct, and it was **slower** when first measured, for the same reason as
// the integer clamp below: `fmax.s` sits on the dependency chain where the
// branch it replaced hung off it and was predicted not-taken. Two builds
// alternated in one board session, `densenet121` 5073.15 / 5071.27 ms against
// 5221.06 -- 2.9% slower, against a 0.2% spread.
//
// **Re-measured a day later, and it wins.** In between, the loop around it lost
// three steps of dependency chain -- the accumulator came out of memory, the
// windows were straightened, the per-channel affine became one multiply-add --
// and the body went from two elements to four. With four short chains
// interleaved there is slack to hide an `fmax.s` in and a mispredicting branch
// to save on every one: `densenet121` **853.28 -> 820.95 ms (-3.8%)**, the whole
// set 2845.76 -> 2813.30 (-1.1%), every other model within +-0.3% and all
// byte-identical. So it is on by default now, and this case says so.
//
// The number that was right about one loop shape was wrong about the next one.
// CHECK-LABEL: func @relu_below_a_quantization
// CHECK:         arith.maxnumf
// CHECK-NOT:     arith.select
func.func @relu_below_a_quantization(%in: memref<8x16xf32>, %m: memref<16xf32>,
                                     %s: memref<16xf32>, %out: memref<8x16xi8>) {
  %zero = arith.constant 0.000000e+00 : f32
  %sc = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                   affine_map<(d0, d1) -> (d1)>,
                                   affine_map<(d0, d1) -> (d1)>,
                                   affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in, %m, %s : memref<8x16xf32>, memref<16xf32>, memref<16xf32>)
      outs(%out : memref<8x16xi8>) {
  ^bb0(%x: f32, %mean: f32, %inv: f32, %o: i8):
    %d = arith.subf %x, %mean : f32
    %n = arith.mulf %d, %inv : f32
    %b = arith.addf %n, %zero : f32
    %p = arith.cmpf ugt, %b, %zero : f32
    %r = arith.select %p, %b, %zero : f32
    %q = arith.mulf %r, %sc : f32
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

// The same relu, but the result is the function's answer: a NaN here is
// something a caller can see, so the NaN rule has to be proved rather than
// argued away. Left alone.
// CHECK-LABEL: func @relu_that_is_the_answer
// CHECK:         arith.select
// CHECK-NOT:     arith.maxnumf
func.func @relu_that_is_the_answer(%in: memref<8x16xf32>, %m: memref<16xf32>,
                                   %s: memref<16xf32>, %out: memref<8x16xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                   affine_map<(d0, d1) -> (d1)>,
                                   affine_map<(d0, d1) -> (d1)>,
                                   affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in, %m, %s : memref<8x16xf32>, memref<16xf32>, memref<16xf32>)
      outs(%out : memref<8x16xf32>) {
  ^bb0(%x: f32, %mean: f32, %inv: f32, %o: f32):
    %d = arith.subf %x, %mean : f32
    %n = arith.mulf %d, %inv : f32
    %p = arith.cmpf ugt, %n, %zero : f32
    %r = arith.select %p, %n, %zero : f32
    linalg.yield %r : f32
  }
  return
}

// -----

// **Tried twice and reverted twice.** The integer clamp at the end of a
// quantization tail can be moved into the float: this board's Rocket is plain
// `rv64gc`, so `arith.maxsi`/`arith.minsi` are a compare, a branch and an `li`
// each, where `fmax.s`/`fmin.s` are one instruction and never branch.
//
// It is exact -- clamping commutes with rounding when the bounds are integers,
// and `scripts/clamp_check.c` checks that over 2,267,807,744 f32 values -- and
// it is **slower**, because an in-order single-issue core pays for the
// dependency chain and the branches hung off it. Measured on the U280, every
// model 0 of 40 against the CPU reference: `gmin` 10.64 -> 11.53 ms,
// `resnet18` 68.49 -> 73.06, `lstm` 18.29 -> 18.55. An earlier session measured
// the same rewrite across 26 models at 225.8 -> 231.1 ms.
//
// So the clamp stays as it is, and the check below is what says so.
// CHECK-LABEL: func @the_integer_clamp_stays
// CHECK:         arith.maxsi
// CHECK:         arith.minsi
// A float clamp would need a minimum as well as a maximum; a relu below it is
// a different rewrite and is allowed.
// CHECK-NOT:     arith.minnumf
func.func @the_integer_clamp_stays(%in: memref<8x16xf32>, %out: memref<8x16xi8>) {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                   affine_map<(d0, d1) -> (d0, d1)>],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in : memref<8x16xf32>) outs(%out : memref<8x16xi8>) {
  ^bb0(%x: f32, %o: i8):
    %q = arith.mulf %x, %s : f32
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

// RUN: gemmlir-opt --select-to-minmax=license-by-destination=1 %s | FileCheck %s --check-prefix=DEST

// **Licensing the rewrite by where the result goes.** The two forms differ only
// on a NaN, and where the value reaches nothing but a conversion to an integer
// that difference is unobservable: `fptosi` of a NaN is poison already, so
// handing back the other operand is a refinement. It is the same licence
// `--saturate-constant-casts` uses on a padding constant.
//
// **This was measured and reverted once, and re-measured.** The first number was
// taken when the loop around it was much longer -- the reduction accumulator
// still went through memory, the windows were still loops, the per-channel
// affine was still three operations an element, and the body held two elements.
// On the chains those left, and with four of them interleaved, there is slack to
// hide an `fmax.s` in and a mispredicting branch to save on every one.
//
// DenseNet's `forward` goes from 40 `fmax.s` to 2428.
// DEST-LABEL: func.func @relu_into_an_integer
// DEST-NOT:     arith.select
// DEST:         arith.maxnumf
func.func @relu_into_an_integer(%x: memref<64xf32>, %out: memref<64xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %z = arith.constant 0.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  scf.for %i = %c0 to %c64 step %c1 {
    %v = memref.load %x[%i] : memref<64xf32>
    %p = arith.cmpf ugt, %v, %z : f32
    %r = arith.select %p, %v, %z : f32
    %rd = math.roundeven %r : f32
    %n = arith.fptosi %rd : f32 to i32
    %a = arith.maxsi %n, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    memref.store %t, %out[%i] : memref<64xi8>
  }
  return
}

// -----

// RUN: gemmlir-opt --select-to-minmax=license-by-destination=1 %s | FileCheck %s --check-prefix=KEEPS

// A result that keeps its f32 answer is observable on a NaN, so the licence does
// not apply and the select stays.
// KEEPS-LABEL: func.func @relu_into_an_f32
// KEEPS:         arith.select
// KEEPS-NOT:     arith.maxnumf
func.func @relu_into_an_f32(%x: memref<64xf32>, %out: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %z = arith.constant 0.000000e+00 : f32
  scf.for %i = %c0 to %c64 step %c1 {
    %v = memref.load %x[%i] : memref<64xf32>
    %p = arith.cmpf ugt, %v, %z : f32
    %r = arith.select %p, %v, %z : f32
    memref.store %r, %out[%i] : memref<64xf32>
  }
  return
}
