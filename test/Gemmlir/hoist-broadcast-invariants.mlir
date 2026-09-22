// RUN: gemmlir-opt --hoist-broadcast-invariants --split-input-file %s | FileCheck %s

// A layer norm's inner loop multiplies a per-row reciprocal by a constant and
// then by the element. The first multiply is the same number for all 64
// channels, so the loop does it 64 times -- and nothing else in the pipeline
// takes it: the constants are already settled, so there is no scaling left to
// move, and `llc` runs no IR pipeline of its own.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @per_row_multiply
// One small loop over the eight rows:
// CHECK:       %[[S:.*]] = memref.alloc() {{.*}} memref<8xf32>
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel"]
// CHECK-SAME:    outs(%[[S]]
// CHECK:         arith.mulf
// and the element loop reads it instead of recomputing it:
// CHECK:       linalg.generic
// CHECK-SAME:    %[[S]]
// CHECK:         arith.subf
// CHECK-NEXT:    arith.mulf
// CHECK-NEXT:    math.roundeven
func.func @per_row_multiply(%x: memref<8x64xf32>, %mean: memref<8xf32>,
                            %rstd: memref<8xf32>, %out: memref<8x64xi8>) {
  %k = arith.constant 2.500000e-01 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id2, #row, #row, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x, %mean, %rstd : memref<8x64xf32>, memref<8xf32>, memref<8xf32>)
      outs(%out : memref<8x64xi8>) {
  ^bb0(%in: f32, %m: f32, %r: f32, %o: i8):
    %s = arith.mulf %r, %k : f32
    %c = arith.subf %in, %m : f32
    %n = arith.mulf %c, %s : f32
    %e = math.roundeven %n : f32
    %i = arith.fptosi %e : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// A chain of per-row work moves as one, and only the value the element loop
// still needs comes back through a buffer -- `rsqrt` here feeds the multiply
// and nothing else, so it costs no buffer of its own.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @a_chain_moves_as_one
// CHECK:       memref.alloc() {{.*}} memref<8xf32>
// CHECK:       linalg.generic
// CHECK-SAME:    iterator_types = ["parallel"]
// CHECK:         math.rsqrt
// CHECK-NEXT:    arith.mulf
// CHECK:       linalg.generic
// CHECK-NOT:     math.rsqrt
// CHECK:         arith.subf
// CHECK-NEXT:    arith.mulf
func.func @a_chain_moves_as_one(%x: memref<8x64xf32>, %mean: memref<8xf32>,
                                %var: memref<8xf32>, %out: memref<8x64xf32>) {
  %k = arith.constant 2.500000e-01 : f32
  linalg.generic {indexing_maps = [#id2, #row, #row, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x, %mean, %var : memref<8x64xf32>, memref<8xf32>, memref<8xf32>)
      outs(%out : memref<8x64xf32>) {
  ^bb0(%in: f32, %m: f32, %v: f32, %o: f32):
    %r = math.rsqrt %v : f32
    %s = arith.mulf %r, %k : f32
    %c = arith.subf %in, %m : f32
    %n = arith.mulf %c, %s : f32
    linalg.yield %n : f32
  }
  return
}

// -----

// Nothing is invariant when every operand varies with the loop.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @nothing_to_hoist
// CHECK-NOT:   memref.alloc
// CHECK:       linalg.generic
// CHECK:         arith.mulf
// CHECK-NEXT:    arith.addf
func.func @nothing_to_hoist(%x: memref<8x64xf32>, %y: memref<8x64xf32>,
                            %out: memref<8x64xf32>) {
  %k = arith.constant 2.500000e-01 : f32
  linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x, %y : memref<8x64xf32>, memref<8x64xf32>)
      outs(%out : memref<8x64xf32>) {
  ^bb0(%in: f32, %in2: f32, %o: f32):
    %m = arith.mulf %in, %k : f32
    %a = arith.addf %m, %in2 : f32
    linalg.yield %a : f32
  }
  return
}
