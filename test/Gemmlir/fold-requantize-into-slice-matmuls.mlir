// RUN: gemmlir-opt --fold-requantize-into-slice-matmuls --split-input-file %s | FileCheck %s

// The attention output is one matmul per head writing its own slice of a shared
// i32 accumulator, and one requantization that reads all of it and puts the
// heads back where the model wants them. The conversion's per-slice fold wants
// the destination laid out exactly like the accumulator, so the permutation
// stops it -- and the loop that is left writes its result with a stride, which
// is the expensive way round.
//
// A slice has one destination block and the runtime addresses a block through a
// row stride, so each call writes its own i8 and the reader goes away.

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#heads = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @attention_output
// Nothing is left of the accumulator or the two loops over it.
// CHECK-NOT:   gemmlir.matmul_i8(
// CHECK-NOT:   arith.sitofp
// CHECK-NOT:   math.roundeven
// Each call writes its own block of the result: 2.0 * 0.25 is the scale.
// CHECK-DAG:   %[[S0:.*]] = memref.subview %arg2[0, 0, 0, 0] [1, 4, 1, 3] [1, 1, 1, 1]
// CHECK-DAG:   %[[S1:.*]] = memref.subview %arg2[0, 0, 1, 0] [1, 4, 1, 3] [1, 1, 1, 1]
// CHECK-DAG:   gemmlir.matmul_i8_scale(%arg0, %arg1, %[[S0]]) {{.*}}scale = 5.000000e-01 : f32}
func.func @attention_output(%a: memref<4x5xi8>, %b: memref<5x3xi8>,
                            %out: memref<1x4x2x3xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %deq = arith.constant 2.0 : f32
  %req = arith.constant 4.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<2x4x3xi32>
  %s0 = memref.subview %acc[%c0, 0, 0] [1, 4, 3] [1, 1, 1] : memref<2x4x3xi32> to memref<4x3xi32, strided<[3, 1], offset: ?>>
  gemmlir.matmul_i8(%a, %b, %s0) : (memref<4x5xi8> x memref<5x3xi8>) -> memref<4x3xi32, strided<[3, 1], offset: ?>> {accumulate = false}
  // The unrolled loop leaves the second offset as arithmetic on the induction
  // variable, not as a constant, and nothing canonicalizes between there and
  // here.
  %step = arith.muli %c1, %c1 : index
  %off1 = arith.addi %c0, %step : index
  %s1 = memref.subview %acc[%off1, 0, 0] [1, 4, 3] [1, 1, 1] : memref<2x4x3xi32> to memref<4x3xi32, strided<[3, 1], offset: ?>>
  gemmlir.matmul_i8(%a, %b, %s1) : (memref<4x5xi8> x memref<5x3xi8>) -> memref<4x3xi32, strided<[3, 1], offset: ?>> {accumulate = false}
  %mid = memref.alloc() : memref<2x4x3xf32>
  linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%acc : memref<2x4x3xi32>) outs(%mid : memref<2x4x3xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %deq : f32
    linalg.yield %m : f32
  }
  %e = memref.expand_shape %mid [[0, 1], [2], [3]] output_shape [1, 2, 4, 3] : memref<2x4x3xf32> into memref<1x2x4x3xf32>
  linalg.generic {indexing_maps = [#heads, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%e : memref<1x2x4x3xf32>) outs(%out : memref<1x4x2x3xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %req : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c = arith.maxsi %i, %lo : i32
    %c2 = arith.minsi %c, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %mid : memref<2x4x3xf32>
  memref.dealloc %acc : memref<2x4x3xi32>
  return
}

// -----

// A permutation that also swaps the matmul's own two axes is not a block the
// runtime can address: the slice would come out 3x4 where the call writes 4x3.

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#swap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @transposed_block_refused
// CHECK:       gemmlir.matmul_i8(
// CHECK-NOT:   gemmlir.matmul_i8_scale(
func.func @transposed_block_refused(%a: memref<4x5xi8>, %b: memref<5x3xi8>,
                                    %out: memref<1x2x3x4xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %deq = arith.constant 2.0 : f32
  %req = arith.constant 4.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<2x4x3xi32>
  %s0 = memref.subview %acc[%c0, 0, 0] [1, 4, 3] [1, 1, 1] : memref<2x4x3xi32> to memref<4x3xi32, strided<[3, 1], offset: ?>>
  gemmlir.matmul_i8(%a, %b, %s0) : (memref<4x5xi8> x memref<5x3xi8>) -> memref<4x3xi32, strided<[3, 1], offset: ?>> {accumulate = false}
  %s1 = memref.subview %acc[%c1, 0, 0] [1, 4, 3] [1, 1, 1] : memref<2x4x3xi32> to memref<4x3xi32, strided<[3, 1], offset: ?>>
  gemmlir.matmul_i8(%a, %b, %s1) : (memref<4x5xi8> x memref<5x3xi8>) -> memref<4x3xi32, strided<[3, 1], offset: ?>> {accumulate = false}
  %mid = memref.alloc() : memref<2x4x3xf32>
  linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%acc : memref<2x4x3xi32>) outs(%mid : memref<2x4x3xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %deq : f32
    linalg.yield %m : f32
  }
  %e = memref.expand_shape %mid [[0, 1], [2], [3]] output_shape [1, 2, 4, 3] : memref<2x4x3xf32> into memref<1x2x4x3xf32>
  linalg.generic {indexing_maps = [#swap, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%e : memref<1x2x4x3xf32>) outs(%out : memref<1x2x3x4xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %req : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c = arith.maxsi %i, %lo : i32
    %c2 = arith.minsi %c, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %mid : memref<2x4x3xf32>
  memref.dealloc %acc : memref<2x4x3xi32>
  return
}

// -----

// One call leaves the other slice undefined, so there is nothing to fold into.

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @one_writer_refused
// CHECK:       gemmlir.matmul_i8(
// CHECK-NOT:   gemmlir.matmul_i8_scale(
func.func @one_writer_refused(%a: memref<4x5xi8>, %b: memref<5x3xi8>,
                              %out: memref<2x4x3xi8>) {
  %c0 = arith.constant 0 : index
  %deq = arith.constant 2.0 : f32
  %req = arith.constant 4.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<2x4x3xi32>
  %s0 = memref.subview %acc[%c0, 0, 0] [1, 4, 3] [1, 1, 1] : memref<2x4x3xi32> to memref<4x3xi32, strided<[3, 1], offset: ?>>
  gemmlir.matmul_i8(%a, %b, %s0) : (memref<4x5xi8> x memref<5x3xi8>) -> memref<4x3xi32, strided<[3, 1], offset: ?>> {accumulate = false}
  %mid = memref.alloc() : memref<2x4x3xf32>
  linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%acc : memref<2x4x3xi32>) outs(%mid : memref<2x4x3xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %deq : f32
    linalg.yield %m : f32
  }
  linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%mid : memref<2x4x3xf32>) outs(%out : memref<2x4x3xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %req : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c = arith.maxsi %i, %lo : i32
    %c2 = arith.minsi %c, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %mid : memref<2x4x3xf32>
  memref.dealloc %acc : memref<2x4x3xi32>
  return
}
