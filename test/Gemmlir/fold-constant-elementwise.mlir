// RUN: gemmlir-opt --fold-constant-elementwise --split-input-file %s | FileCheck %s

// A per-channel loop that reads nothing but constant globals computes the same
// numbers on every inference. `--hoist-invariant-reciprocal` and
// `--combine-channel-affine` both write one, and both run in MID -- long after
// the constant folding in FRONT stopped looking.
//
// DenseNet-121 runs 31,616 steps of them, and `fsqrt.s` with `fdiv.s` is 5.3%
// of that model by program-counter sampling.

memref.global "private" constant @var : memref<4xf32> = dense<[4.0, 9.0, 16.0, 25.0]>

// 1/sqrt of 4, 9, 16 and 25.
// CHECK-DAG: memref.global "private" constant @__gemmlir_folded_0 : memref<4xf32> = dense<[5.000000e-01, 0.333333343, 2.500000e-01, 2.000000e-01]>

// CHECK-LABEL: func.func @a_rsqrt_of_a_constant
// CHECK-NOT:   math.rsqrt
// CHECK-NOT:   memref.alloc
// CHECK:       %[[G:.*]] = memref.get_global @__gemmlir_folded_0
// CHECK:       linalg.generic {{.*}} ins(%[[G]]
func.func @a_rsqrt_of_a_constant(%dst: memref<4xf32>) {
  %v = memref.get_global @var : memref<4xf32>
  %out = memref.alloc() : memref<4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%v : memref<4xf32>) outs(%out : memref<4xf32>) {
  ^bb0(%in: f32, %o: f32):
    %r = math.rsqrt %in : f32
    linalg.yield %r : f32
  }
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%out : memref<4xf32>) outs(%dst : memref<4xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  }
  memref.dealloc %out : memref<4xf32>
  return
}

// -----

// The epsilon of a batch norm is an `arith.constant` *outside* the region with
// an `arith.truncf` in, so an operand that is neither a block argument nor
// another body result is the rule rather than the exception.

memref.global "private" constant @b_var : memref<2xf32> = dense<[3.0, 8.0]>

// 1/sqrt(3+1) and 1/sqrt(8+1).
// CHECK-DAG: memref.global "private" constant @__gemmlir_folded_0 : memref<2xf32> = dense<[5.000000e-01, 0.333333343]>

// CHECK-LABEL: func.func @b_epsilon_from_outside
// CHECK-NOT:   math.rsqrt
// CHECK:       memref.get_global @__gemmlir_folded_0
func.func @b_epsilon_from_outside(%dst: memref<2xf32>) {
  %eps = arith.constant 1.000000e+00 : f64
  %v = memref.get_global @b_var : memref<2xf32>
  %out = memref.alloc() : memref<2xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%v : memref<2xf32>) outs(%out : memref<2xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = arith.truncf %eps : f64 to f32
    %s = arith.addf %in, %e : f32
    %r = math.rsqrt %s : f32
    linalg.yield %r : f32
  }
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%out : memref<2xf32>) outs(%dst : memref<2xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  }
  memref.dealloc %out : memref<2xf32>
  return
}

// -----

// Two outputs from one loop, and the second loop reads the first one's buffer:
// the greedy driver comes back round once that buffer is a constant global.
// This is the pair `--hoist-invariant-reciprocal` and
// `--combine-channel-affine` leave behind, in miniature.

memref.global "private" constant @c_var : memref<2xf32> = dense<[4.0, 16.0]>
memref.global "private" constant @c_mean : memref<2xf32> = dense<[1.0, 2.0]>

// rsqrt(4)=0.5 and rsqrt(16)=0.25, then -(1*0.5) and -(2*0.25).
// CHECK-DAG: memref.global "private" constant @{{.*}} : memref<2xf32> = dense<[5.000000e-01, 2.500000e-01]>
// CHECK-DAG: memref.global "private" constant @{{.*}} : memref<2xf32> = dense<-5.000000e-01>

// CHECK-LABEL: func.func @c_chains_through
// CHECK-NOT:   linalg.generic
// CHECK-NOT:   memref.alloc
// CHECK:       return
func.func @c_chains_through(%da: memref<2xf32>, %db: memref<2xf32>) {
  %v = memref.get_global @c_var : memref<2xf32>
  %m = memref.get_global @c_mean : memref<2xf32>
  %r = memref.alloc() : memref<2xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%v : memref<2xf32>) outs(%r : memref<2xf32>) {
  ^bb0(%in: f32, %o: f32):
    %q = math.rsqrt %in : f32
    linalg.yield %q : f32
  }
  %a = memref.alloc() : memref<2xf32>
  %b = memref.alloc() : memref<2xf32>
  %zero = arith.constant 0.0 : f32
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                                   affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%m, %r : memref<2xf32>, memref<2xf32>) outs(%a, %b : memref<2xf32>, memref<2xf32>) {
  ^bb0(%mi: f32, %ri: f32, %oa: f32, %ob: f32):
    %p = arith.mulf %mi, %ri : f32
    %n = arith.subf %zero, %p : f32
    linalg.yield %ri, %n : f32, f32
  }
  memref.copy %a, %da : memref<2xf32> to memref<2xf32>
  memref.copy %b, %db : memref<2xf32> to memref<2xf32>
  memref.dealloc %r : memref<2xf32>
  memref.dealloc %a : memref<2xf32>
  memref.dealloc %b : memref<2xf32>
  return
}

// -----

// A buffer something else writes is not a constant: a `linalg.fill` after the
// loop means the bytes the next reader sees are not the ones evaluated.

memref.global "private" constant @d_var : memref<2xf32> = dense<[4.0, 9.0]>

// CHECK-LABEL: func.func @d_refuses_a_second_writer
// CHECK:       math.rsqrt
func.func @d_refuses_a_second_writer() {
  %v = memref.get_global @d_var : memref<2xf32>
  %out = memref.alloc() : memref<2xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%v : memref<2xf32>) outs(%out : memref<2xf32>) {
  ^bb0(%in: f32, %o: f32):
    %r = math.rsqrt %in : f32
    linalg.yield %r : f32
  }
  %c = arith.constant 0.0 : f32
  linalg.fill ins(%c : f32) outs(%out : memref<2xf32>)
  memref.dealloc %out : memref<2xf32>
  return
}

// -----

// A `memref.subview` is a second name for the same bytes and has no memory
// effects to read, so it is refused along with everything else unrecognised.

memref.global "private" constant @e_var : memref<2xf32> = dense<[4.0, 9.0]>

// CHECK-LABEL: func.func @e_refuses_an_alias
// CHECK:       math.rsqrt
func.func @e_refuses_an_alias() -> memref<1xf32, strided<[1]>> {
  %v = memref.get_global @e_var : memref<2xf32>
  %out = memref.alloc() : memref<2xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%v : memref<2xf32>) outs(%out : memref<2xf32>) {
  ^bb0(%in: f32, %o: f32):
    %r = math.rsqrt %in : f32
    linalg.yield %r : f32
  }
  %s = memref.subview %out[0] [1] [1] : memref<2xf32> to memref<1xf32, strided<[1]>>
  return %s : memref<1xf32, strided<[1]>>
}

// -----

// An operation the evaluator does not know refuses rather than guesses: a wrong
// constant is silent. `math.exp` is not correctly rounded, so it never joins.

memref.global "private" constant @f_var : memref<2xf32> = dense<[1.0, 2.0]>

// CHECK-LABEL: func.func @f_refuses_an_unknown_body
// CHECK:       math.exp
func.func @f_refuses_an_unknown_body(%dst: memref<2xf32>) {
  %v = memref.get_global @f_var : memref<2xf32>
  %out = memref.alloc() : memref<2xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%v : memref<2xf32>) outs(%out : memref<2xf32>) {
  ^bb0(%in: f32, %o: f32):
    %r = math.exp %in : f32
    linalg.yield %r : f32
  }
  memref.copy %out, %dst : memref<2xf32> to memref<2xf32>
  memref.dealloc %out : memref<2xf32>
  return
}

// -----

// A buffer that reads what it already holds is not a function of its inputs; a
// fresh `memref.alloc` holds nothing to read.

memref.global "private" constant @h_var : memref<2xf32> = dense<[1.0, 2.0]>

// CHECK-LABEL: func.func @h_refuses_an_accumulator
// CHECK:       linalg.generic
func.func @h_refuses_an_accumulator(%dst: memref<2xf32>) {
  %v = memref.get_global @h_var : memref<2xf32>
  %out = memref.alloc() : memref<2xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%v : memref<2xf32>) outs(%out : memref<2xf32>) {
  ^bb0(%in: f32, %o: f32):
    %r = arith.addf %in, %o : f32
    linalg.yield %r : f32
  }
  memref.copy %out, %dst : memref<2xf32> to memref<2xf32>
  memref.dealloc %out : memref<2xf32>
  return
}

// -----

// A buffer bigger than `max-elements` stays a loop: the constant would go in
// the object, and a feature map does not belong there.

memref.global "private" constant @g_var : memref<64xf32> = dense<4.0>

// CHECK-LABEL: func.func @g_folds_by_default
// CHECK-NOT:   math.rsqrt
func.func @g_folds_by_default(%dst: memref<64xf32>) {
  %v = memref.get_global @g_var : memref<64xf32>
  %out = memref.alloc() : memref<64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%v : memref<64xf32>) outs(%out : memref<64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %r = math.rsqrt %in : f32
    linalg.yield %r : f32
  }
  memref.copy %out, %dst : memref<64xf32> to memref<64xf32>
  memref.dealloc %out : memref<64xf32>
  return
}

// RUN: gemmlir-opt --fold-constant-elementwise=max-elements=8 --split-input-file %s | FileCheck %s --check-prefix=SMALL
// SMALL-LABEL: func.func @g_folds_by_default
// SMALL:       math.rsqrt
