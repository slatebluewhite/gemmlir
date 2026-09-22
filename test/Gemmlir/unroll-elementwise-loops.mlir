// RUN: gemmlir-opt --unroll-elementwise-loops %s | FileCheck %s
// RUN: gemmlir-opt --unroll-elementwise-loops=factor=2 %s | FileCheck %s --check-prefix=TWO

// The innermost loop of an f32/i8 conversion is a chain this in-order core
// walks with nothing to put in the gaps. More iterations in one body give the
// scheduler more independent chains to interleave.
//
// `factor` is a **ceiling**, not the only candidate: it is halved until it
// divides the trip count, so there is never an epilogue and a large ceiling
// still reaches every loop a small one would.

// CHECK-LABEL: func.func @leaf
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %[[S:.*]] {
// CHECK-COUNT-8:   memref.store
// CHECK-NOT:       memref.store
// CHECK:         }
// TWO-LABEL:  func.func @leaf
// TWO-COUNT-2:   memref.store
// TWO:         }
func.func @leaf(%in: memref<64xf32>, %out: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %two = arith.constant 2.0 : f32
  scf.for %i = %c0 to %c64 step %c1 {
    %v = memref.load %in[%i] : memref<64xf32>
    %m = arith.mulf %v, %two : f32
    memref.store %m, %out[%i] : memref<64xf32>
  }
  return
}

// A loop with another loop inside it is the nest's scaffolding; unrolling it
// duplicates the whole subtree for no gain.
// CHECK-LABEL: func.func @outer_stays
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK-COUNT-8:     memref.store
// CHECK-NOT:         memref.store
func.func @outer_stays(%in: memref<8x64xf32>, %out: memref<8x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %two = arith.constant 2.0 : f32
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c64 step %c1 {
      %v = memref.load %in[%i, %j] : memref<8x64xf32>
      %m = arith.mulf %v, %two : f32
      memref.store %m, %out[%i, %j] : memref<8x64xf32>
    }
  }
  return
}

// A loop with a call in it is an accelerator or a copy, whose cost is not the
// loop arithmetic.
// CHECK-LABEL: func.func @calls_out
// CHECK:         scf.for
// CHECK-COUNT-1:   func.call
// CHECK-NOT:       func.call
func.func private @work(%i: index)
func.func @calls_out() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c64 step %c1 {
    func.call @work(%i) : (index) -> ()
  }
  return
}

// An odd trip count would need an epilogue -- a second copy of the body for a
// remainder that does not happen in this pipeline.
// CHECK-LABEL: func.func @odd_trip
// CHECK:         scf.for
// CHECK-COUNT-1:   memref.store
// CHECK-NOT:       memref.store
func.func @odd_trip(%in: memref<15xf32>, %out: memref<15xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c15 = arith.constant 15 : index
  %two = arith.constant 2.0 : f32
  scf.for %i = %c0 to %c15 step %c1 {
    %v = memref.load %in[%i] : memref<15xf32>
    %m = arith.mulf %v, %two : f32
    memref.store %m, %out[%i] : memref<15xf32>
  }
  return
}

// A loop shorter than two unrolled bodies has nothing to interleave.
// CHECK-LABEL: func.func @too_short
// CHECK:         scf.for
// CHECK-COUNT-1:   memref.store
// CHECK-NOT:       memref.store
func.func @too_short(%in: memref<2xf32>, %out: memref<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %two = arith.constant 2.0 : f32
  scf.for %i = %c0 to %c2 step %c1 {
    %v = memref.load %in[%i] : memref<2xf32>
    %m = arith.mulf %v, %two : f32
    memref.store %m, %out[%i] : memref<2xf32>
  }
  return
}

// Bounds that are not constants here: nothing says the factor divides them.
// CHECK-LABEL: func.func @dynamic_bound
// CHECK:         scf.for
// CHECK-COUNT-1:   memref.store
// CHECK-NOT:       memref.store
func.func @dynamic_bound(%in: memref<?xf32>, %out: memref<?xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %two = arith.constant 2.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    %v = memref.load %in[%i] : memref<?xf32>
    %m = arith.mulf %v, %two : f32
    memref.store %m, %out[%i] : memref<?xf32>
  }
  return
}

// Twelve does not divide by eight, so the ceiling comes down to four. Without
// the halving this loop would not be unrolled at all.
// CHECK-LABEL: func.func @twelve_takes_four
// CHECK:         scf.for
// CHECK-COUNT-4:   memref.store
// CHECK-NOT:       memref.store
func.func @twelve_takes_four(%in: memref<12xf32>, %out: memref<12xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %two = arith.constant 2.0 : f32
  scf.for %i = %c0 to %c12 step %c1 {
    %v = memref.load %in[%i] : memref<12xf32>
    %m = arith.mulf %v, %two : f32
    memref.store %m, %out[%i] : memref<12xf32>
  }
  return
}

// Six divides by neither eight nor four, and two is still better than one.
// CHECK-LABEL: func.func @six_takes_two
// CHECK:         scf.for
// CHECK-COUNT-2:   memref.store
// CHECK-NOT:       memref.store
func.func @six_takes_two(%in: memref<6xf32>, %out: memref<6xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %two = arith.constant 2.0 : f32
  scf.for %i = %c0 to %c6 step %c1 {
    %v = memref.load %in[%i] : memref<6xf32>
    %m = arith.mulf %v, %two : f32
    memref.store %m, %out[%i] : memref<6xf32>
  }
  return
}

// Eight divides sixteen, but a loop has to be twice the unrolled body for the
// unrolling to be worth the code: sixteen trips take eight, twelve take four,
// and eight trips come down to four as well.
// CHECK-LABEL: func.func @eight_trips_take_four
// CHECK:         scf.for
// CHECK-COUNT-4:   memref.store
// CHECK-NOT:       memref.store
func.func @eight_trips_take_four(%in: memref<8xf32>, %out: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %two = arith.constant 2.0 : f32
  scf.for %i = %c0 to %c8 step %c1 {
    %v = memref.load %in[%i] : memref<8xf32>
    %m = arith.mulf %v, %two : f32
    memref.store %m, %out[%i] : memref<8xf32>
  }
  return
}
