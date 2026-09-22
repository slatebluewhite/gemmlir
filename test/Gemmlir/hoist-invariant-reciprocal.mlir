// RUN: gemmlir-opt --split-input-file --hoist-invariant-reciprocal %s | FileCheck %s

// A softmax divides every element of a row by that row's sum. The divisor is
// read through a map that drops the inner loop, so one reciprocal serves the
// whole row -- 64 divisions instead of 4096.

#row  = affine_map<(d0, d1) -> (d0)>
#full = affine_map<(d0, d1) -> (d0, d1)>
#one  = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @softmax_tail
// The reciprocal loop: one over the divisor, kept finite.
// CHECK:         %[[BIG:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK:         %[[R:.*]] = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel"]
// CHECK-SAME:      ins(%[[SUM:.*]] : memref<64xf32>) outs(%[[R]] : memref<64xf32>)
// CHECK:           %[[INV:.*]] = arith.divf %{{.*}}, %[[IN:.*]] : f32
// CHECK:           %[[MAG:.*]] = math.absf %[[INV]]
// CHECK:           %[[OK:.*]] = arith.cmpf ole, %[[MAG]], %[[BIG]]
// CHECK:           %[[CLAMP:.*]] = math.copysign %[[BIG]], %[[INV]]
// CHECK:           arith.select %[[OK]], %[[INV]], %[[CLAMP]]
// And the big loop multiplies by it. The remaining divide is by the
// quantization's own constant scale, which --reciprocal-for-division folds.
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %[[R]] :
// CHECK:         ^bb0(%[[E:.*]]: f32, %[[RR:.*]]: f32, %{{.*}}: i8):
// CHECK:           arith.mulf %[[E]], %[[RR]] : f32
// CHECK:           arith.fptosi
// CHECK:         memref.dealloc %[[R]]
func.func @softmax_tail(%exp: memref<64x64xf32>, %sum: memref<64xf32>,
                        %out: memref<64x64xi8>) {
  %scale = arith.constant 0.0078125 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#full, #row, #full],
                  iterator_types = ["parallel", "parallel"]}
      ins(%exp, %sum : memref<64x64xf32>, memref<64xf32>) outs(%out : memref<64x64xi8>) {
  ^bb0(%e: f32, %s: f32, %o: i8):
    %p = arith.divf %e, %s : f32
    %q = arith.divf %p, %scale : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %b = arith.trunci %c1 : i32 to i8
    linalg.yield %b : i8
  }
  return
}

// -----

// The quotient has to reach a conversion to an integer. That is what makes the
// runtime reciprocal's edge cases unobservable -- where the clamp bites, the
// quotient was already outside the integer's range. An f32 result keeps the
// division.

#row  = affine_map<(d0, d1) -> (d0)>
#full = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @stays_float
// CHECK-NOT:     memref.alloc
// CHECK:         arith.divf
func.func @stays_float(%in: memref<8x16xf32>, %d: memref<8xf32>,
                       %out: memref<8x16xf32>) {
  linalg.generic {indexing_maps = [#full, #row, #full],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in, %d : memref<8x16xf32>, memref<8xf32>) outs(%out : memref<8x16xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %q = arith.divf %a, %b : f32
    linalg.yield %q : f32
  }
  return
}

// -----

// A divisor read with the full map is not invariant in anything -- one
// reciprocal per element would be the same number of divides plus a buffer.

#full = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @not_invariant
// CHECK-NOT:     memref.alloc
// CHECK:         arith.divf
func.func @not_invariant(%in: memref<8x16xf32>, %d: memref<8x16xf32>,
                         %out: memref<8x16xi8>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#full, #full, #full],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in, %d : memref<8x16xf32>, memref<8x16xf32>) outs(%out : memref<8x16xi8>) {
  ^bb0(%a: f32, %b: f32, %o: i8):
    %q = arith.divf %a, %b : f32
    %i = arith.fptosi %q : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// An operand that is also read as something other than a divisor stays put:
// rewriting it would leave the row read twice, once for each form.

#row  = affine_map<(d0, d1) -> (d0)>
#full = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @also_an_addend
// CHECK-NOT:     memref.alloc
// CHECK:         arith.divf
func.func @also_an_addend(%in: memref<8x16xf32>, %d: memref<8xf32>,
                          %out: memref<8x16xi8>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#full, #row, #full],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in, %d : memref<8x16xf32>, memref<8xf32>) outs(%out : memref<8x16xi8>) {
  ^bb0(%a: f32, %b: f32, %o: i8):
    %q = arith.divf %a, %b : f32
    %s = arith.addf %q, %b : f32
    %i = arith.fptosi %s : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// The dividend's own operand position matters: `b / a` is not a division by
// the invariant row.

#row  = affine_map<(d0, d1) -> (d0)>
#full = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @row_is_the_dividend
// CHECK-NOT:     memref.alloc
// CHECK:         arith.divf
func.func @row_is_the_dividend(%in: memref<8x16xf32>, %d: memref<8xf32>,
                               %out: memref<8x16xi8>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#full, #row, #full],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in, %d : memref<8x16xf32>, memref<8xf32>) outs(%out : memref<8x16xi8>) {
  ^bb0(%a: f32, %b: f32, %o: i8):
    %q = arith.divf %b, %a : f32
    %i = arith.fptosi %q : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// A batch norm's `rsqrt(var[c] + eps)` depends only on the channel, so it can
// be evaluated once per channel instead of once per element.
//
// `--fold-batch-norm` takes the whole affine into the weights of the
// contraction above it, and where there is one this never sees it. Where there
// is **not** one -- a DenseNet layer's batch norm sits on a join of everything
// the block has produced so far -- the region stays. On `densenet121` that is
// **8,128,512** `rsqrt` evaluations across 593 regions; after this, **34,336**.
//
// What makes it sound is narrower than the reciprocal above and needs none of
// that argument about denormals: `rsqrt(b + c)` is a function of `b` alone, so
// the buffer holds exactly what the body would have computed, bit for bit --
// no reassociation, no reciprocal.
//
// The epsilon reaches the region as an f64 `arith.constant` outside and an
// `arith.truncf` **inside**, so the addend is a body operation that reads
// nothing per-element. It counts as invariant and is cloned into the new loop;
// refusing it is why the pattern first fired on none of the 593.
#chan = affine_map<(d0, d1) -> (d1)>
#both = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @batch_norm_rsqrt_is_per_channel
// One pass over the 16 channels...
// CHECK:         %[[B:.*]] = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg2 : memref<16xf32>)
// CHECK-SAME:      outs(%[[B]] : memref<16xf32>)
// CHECK:           arith.truncf
// CHECK:           arith.addf
// CHECK:           math.rsqrt
// ...and the element loop reads it.
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0, %arg1, %[[B]]
// CHECK-NOT:       math.rsqrt
// CHECK:           arith.subf
// CHECK:           arith.mulf
func.func @batch_norm_rsqrt_is_per_channel(%in: memref<8x16xf32>,
                                           %mean: memref<16xf32>,
                                           %var: memref<16xf32>,
                                           %out: memref<8x16xi8>) {
  %eps = arith.constant 1.000000e-05 : f64
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#both, #chan, #chan, #both],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in, %mean, %var : memref<8x16xf32>, memref<16xf32>, memref<16xf32>)
      outs(%out : memref<8x16xi8>) {
  ^bb0(%x: f32, %m: f32, %v: f32, %o: i8):
    %e = arith.truncf %eps : f64 to f32
    %ve = arith.addf %v, %e : f32
    %r = math.rsqrt %ve : f32
    %d = arith.subf %x, %m : f32
    %n = arith.mulf %d, %r : f32
    %q = arith.divf %n, %s : f32
    %i = arith.fptosi %q : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// An operand that is a divisor *and* under an `rsqrt` is left alone: rewriting
// one of the two roles would leave it read twice for no gain.
#chan = affine_map<(d0, d1) -> (d1)>
#both = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @two_roles_is_left_alone
// CHECK:         linalg.generic
// CHECK:           math.rsqrt
func.func @two_roles_is_left_alone(%in: memref<8x16xf32>, %v: memref<16xf32>,
                                   %out: memref<8x16xi8>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#both, #chan, #both],
                  iterator_types = ["parallel", "parallel"]}
      ins(%in, %v : memref<8x16xf32>, memref<16xf32>) outs(%out : memref<8x16xi8>) {
  ^bb0(%x: f32, %c: f32, %o: i8):
    %r = math.rsqrt %c : f32
    %n = arith.mulf %x, %r : f32
    %q = arith.divf %n, %c : f32
    %i = arith.fptosi %q : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}
