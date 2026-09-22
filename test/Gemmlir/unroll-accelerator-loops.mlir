// RUN: gemmlir-opt --unroll-accelerator-loops --split-input-file %s | FileCheck %s

// `--place-cache-flushes` models a region it cannot otherwise account for as one
// host step that reads and writes everything inside, and never gives the
// accelerator calls in that region an attribute -- so they keep both their
// flushes, and a flush is a walk of the whole L1.
//
// A transformer is the only shape in the set with such loops: a batch matmul
// lowers to one slice a trip, and `vit_tiny` has 24 of them, each three
// iterations of nothing but a `gemmlir.matmul_i8`. Unrolling them puts the
// calls in the straight line the analysis already handles: `no_flush_after`
// goes from 50 of 74 to 122 of 122, and `no_flush_before` from 0 to 48.

// CHECK-LABEL: func.func @a_a_loop_of_matmuls
// CHECK-NOT:   scf.for
// CHECK-COUNT-3: gemmlir.matmul_i8
// CHECK-NOT:   gemmlir.matmul_i8
func.func @a_a_loop_of_matmuls(%lhs: memref<3x17x64xi8>, %rhs: memref<3x64x17xi8>,
                               %out: memref<3x17x17xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %h = %c0 to %c3 step %c1 {
    %a = memref.subview %lhs[%h, 0, 0] [1, 17, 64] [1, 1, 1]
        : memref<3x17x64xi8> to memref<17x64xi8, strided<[64, 1], offset: ?>>
    %b = memref.subview %rhs[%h, 0, 0] [1, 64, 17] [1, 1, 1]
        : memref<3x64x17xi8> to memref<64x17xi8, strided<[17, 1], offset: ?>>
    %c = memref.subview %out[%h, 0, 0] [1, 17, 17] [1, 1, 1]
        : memref<3x17x17xi32> to memref<17x17xi32, strided<[17, 1], offset: ?>>
    gemmlir.matmul_i8(%a, %b, %c)
        : (memref<17x64xi8, strided<[64, 1], offset: ?>> x memref<64x17xi8, strided<[17, 1], offset: ?>>)
        -> memref<17x17xi32, strided<[17, 1], offset: ?>> {accumulate = false}
  }
  return
}

// -----

// One host operation in the body and unrolling buys nothing: the analysis would
// keep the flushes anyway, and the code would only be longer.

#id = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @b_a_host_step_in_the_loop
// CHECK:       scf.for
func.func @b_a_host_step_in_the_loop(%lhs: memref<3x17x64xi8>, %rhs: memref<3x64x17xi8>,
                                     %out: memref<3x17x17xi32>, %side: memref<17x17xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %z = arith.constant 0 : i32
  scf.for %h = %c0 to %c3 step %c1 {
    %a = memref.subview %lhs[%h, 0, 0] [1, 17, 64] [1, 1, 1]
        : memref<3x17x64xi8> to memref<17x64xi8, strided<[64, 1], offset: ?>>
    %b = memref.subview %rhs[%h, 0, 0] [1, 64, 17] [1, 1, 1]
        : memref<3x64x17xi8> to memref<64x17xi8, strided<[17, 1], offset: ?>>
    %c = memref.subview %out[%h, 0, 0] [1, 17, 17] [1, 1, 1]
        : memref<3x17x17xi32> to memref<17x17xi32, strided<[17, 1], offset: ?>>
    gemmlir.matmul_i8(%a, %b, %c)
        : (memref<17x64xi8, strided<[64, 1], offset: ?>> x memref<64x17xi8, strided<[17, 1], offset: ?>>)
        -> memref<17x17xi32, strided<[17, 1], offset: ?>> {accumulate = false}
    linalg.fill ins(%z : i32) outs(%side : memref<17x17xi32>)
  }
  return
}

// -----

// A loop with no accelerator call in it is not this pass's business; unrolling
// host work is `--unroll-elementwise-loops`, with its own measured factor.

// CHECK-LABEL: func.func @c_no_accelerator
// CHECK:       scf.for
func.func @c_no_accelerator(%buf: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %z = arith.constant 0 : i32
  scf.for %h = %c0 to %c3 step %c1 {
    memref.store %z, %buf[%h] : memref<8xi32>
  }
  return
}

// -----

// A trip count past the cap would trade a walk of the L1 for a walk of the
// instruction cache.

// CHECK-LABEL: func.func @d_too_many_trips
// CHECK:       scf.for
func.func @d_too_many_trips(%lhs: memref<64x17x64xi8>, %rhs: memref<64x64x17xi8>,
                            %out: memref<64x17x17xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %h = %c0 to %c64 step %c1 {
    %a = memref.subview %lhs[%h, 0, 0] [1, 17, 64] [1, 1, 1]
        : memref<64x17x64xi8> to memref<17x64xi8, strided<[64, 1], offset: ?>>
    %b = memref.subview %rhs[%h, 0, 0] [1, 64, 17] [1, 1, 1]
        : memref<64x64x17xi8> to memref<64x17xi8, strided<[17, 1], offset: ?>>
    %c = memref.subview %out[%h, 0, 0] [1, 17, 17] [1, 1, 1]
        : memref<64x17x17xi32> to memref<17x17xi32, strided<[17, 1], offset: ?>>
    gemmlir.matmul_i8(%a, %b, %c)
        : (memref<17x64xi8, strided<[64, 1], offset: ?>> x memref<64x17xi8, strided<[17, 1], offset: ?>>)
        -> memref<17x17xi32, strided<[17, 1], offset: ?>> {accumulate = false}
  }
  return
}
