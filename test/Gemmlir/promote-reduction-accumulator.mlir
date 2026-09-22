// RUN: gemmlir-opt --promote-reduction-accumulator %s | FileCheck %s

// `--convert-linalg-to-loops` writes a reduction's running value back to its
// output buffer every step and reads it again on the next one. On this in-order
// core that store-to-load round trip is the loop: sampling GoogLeNet's program
// counter puts 82% of the model in loops of exactly this shape, and `llc` runs
// no IR pipeline, so nothing downstream promotes the slot.
//
// It cascades. Promoting the innermost loop leaves the load and the store in
// the loop above, which is then a candidate itself -- a 3x3 window ends with
// **one** load before the nest and **one** store after it, and both middle
// loops carry the value as an iteration argument.

// CHECK-LABEL: func.func @a_max_pool_window
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               %[[INIT:.*]] = memref.load %arg1
// CHECK:               %[[OUTER:.*]] = scf.for {{.*}} iter_args(%[[A:.*]] = %[[INIT]]) -> (i8)
// CHECK:                 %[[INNER:.*]] = scf.for {{.*}} iter_args(%[[B:.*]] = %[[A]]) -> (i8)
// CHECK:                   memref.load %arg0
// CHECK:                   %[[M:.*]] = arith.maxsi %[[B]]
// CHECK:                   scf.yield %[[M]]
// CHECK:                 scf.yield %[[INNER]]
// CHECK:               memref.store %[[OUTER]], %arg1
// CHECK-NOT:       memref.store
func.func @a_max_pool_window(%in: memref<1x10x10x4xi8>, %out: memref<1x4x4x4xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.for %oh = %c0 to %c4 step %c1 {
    scf.for %ow = %c0 to %c4 step %c1 {
      scf.for %c = %c0 to %c4 step %c1 {
        scf.for %kh = %c0 to %c3 step %c1 {
          scf.for %kw = %c0 to %c3 step %c1 {
            %ih = arith.addi %oh, %kh : index
            %iw = arith.addi %ow, %kw : index
            %v = memref.load %in[%c0, %ih, %iw, %c] : memref<1x10x10x4xi8>
            %a = memref.load %out[%c0, %oh, %ow, %c] : memref<1x4x4x4xi8>
            %m = arith.maxsi %a, %v : i8
            memref.store %m, %out[%c0, %oh, %ow, %c] : memref<1x4x4x4xi8>
          }
        }
      }
    }
  }
  return
}

// -----

// A float accumulation is the same shape and the same rewrite -- this is the
// sum a mean or an average pool reduces with.
// CHECK-LABEL: func.func @a_running_sum
// CHECK:         memref.load %arg1
// CHECK:         scf.for {{.*}} iter_args
// CHECK:           arith.addf
// CHECK:           scf.yield
// CHECK:         memref.store
func.func @a_running_sum(%in: memref<8x16xf32>, %out: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %v = memref.load %in[%i, %j] : memref<8x16xf32>
      %a = memref.load %out[%i] : memref<8xf32>
      %s = arith.addf %a, %v : f32
      memref.store %s, %out[%i] : memref<8xf32>
    }
  }
  return
}

// -----

// The index moves with the loop, so the slot is a different one each step and
// there is nothing to carry.
// CHECK-LABEL: func.func @the_slot_moves
// CHECK:         scf.for
// CHECK-NOT:     iter_args
// CHECK:         memref.store
func.func @the_slot_moves(%in: memref<16xf32>, %out: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %j = %c0 to %c16 step %c1 {
    %v = memref.load %in[%j] : memref<16xf32>
    %a = memref.load %out[%j] : memref<16xf32>
    %s = arith.addf %a, %v : f32
    memref.store %s, %out[%j] : memref<16xf32>
  }
  return
}

// -----

// A store under an `scf.if` writes the slot only on some steps. Hoisting it out
// of the memory would have to invent what the other branch left there, so the
// loop is refused: the store is not in the loop's own body.
// CHECK-LABEL: func.func @a_conditional_update
// CHECK:         scf.for
// CHECK-NOT:     iter_args
// CHECK:         scf.if
// CHECK:           memref.store
func.func @a_conditional_update(%in: memref<16xf32>, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %z = arith.constant 0.000000e+00 : f32
  scf.for %j = %c0 to %c16 step %c1 {
    %v = memref.load %in[%j] : memref<16xf32>
    %a = memref.load %out[%c0] : memref<1xf32>
    %p = arith.cmpf ogt, %v, %z : f32
    scf.if %p {
      %s = arith.addf %a, %v : f32
      memref.store %s, %out[%c0] : memref<1xf32>
    }
  }
  return
}

// -----

// A call in the loop is opaque -- the accelerator, a copy, a memset -- and could
// write the slot itself. Nothing may write memory inside the loop except the one
// store, at any depth.
// CHECK-LABEL: func.func @a_call_in_the_loop
// CHECK:         scf.for
// CHECK-NOT:     iter_args
func.func @a_call_in_the_loop(%in: memref<16xf32>, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %j = %c0 to %c16 step %c1 {
    func.call @sink(%in) : (memref<16xf32>) -> ()
    %v = memref.load %in[%j] : memref<16xf32>
    %a = memref.load %out[%c0] : memref<1xf32>
    %s = arith.addf %a, %v : f32
    memref.store %s, %out[%c0] : memref<1xf32>
  }
  return
}
func.func private @sink(memref<16xf32>)

// -----

// The buffer is read a second time inside the loop, at an index that is not
// obviously a different slot. Keeping the running value in a register would
// leave that read looking at a value nobody is writing any more.
// CHECK-LABEL: func.func @a_second_reader
// CHECK:         scf.for
// CHECK-NOT:     iter_args
func.func @a_second_reader(%in: memref<16xf32>, %out: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  scf.for %j = %c0 to %c16 step %c1 {
    %v = memref.load %in[%j] : memref<16xf32>
    %a = memref.load %out[%c0] : memref<4xf32>
    %o = memref.load %out[%c2] : memref<4xf32>
    %t = arith.addf %a, %v : f32
    %s = arith.mulf %t, %o : f32
    memref.store %s, %out[%c0] : memref<4xf32>
  }
  return
}
