// RUN: gemmlir-opt --unroll-reduction-windows %s | FileCheck %s

// Once `--promote-reduction-accumulator` has taken the store-to-load round trip
// out of a reduction, what is left in the hot loops is almost pure
// `addi`/`li`/`add`/`blt`/`lui` -- the nest scaffolding. A 3x3 max-pool is a
// six-deep nest whose innermost loop runs *three* times, so the bookkeeping
// costs more than the nine loads and eight compares it exists to schedule.
//
// Both window levels go, and what is left is nine loads and eight compares off
// constant offsets, with the accumulator already in a register:
//
//     lb   s1,-1(a2)
//     lb   a0,0(a1)
//     bge  a0,s1,...
//     lb   a0,64(a1)
//     ...

// CHECK-LABEL: func.func @a_three_by_three_window
// CHECK-NOT:     scf.for {{.*}}iter_args
// CHECK-COUNT-9: memref.load %arg0
// CHECK-NOT:     memref.load %arg0
// CHECK:         memref.store
func.func @a_three_by_three_window(%in: memref<1x10x10x4xi8>, %out: memref<1x4x4x4xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.for %oh = %c0 to %c4 step %c1 {
    scf.for %ow = %c0 to %c4 step %c1 {
      scf.for %c = %c0 to %c4 step %c1 {
        %init = memref.load %out[%c0, %oh, %ow, %c] : memref<1x4x4x4xi8>
        %r = scf.for %kh = %c0 to %c3 step %c1 iter_args(%a = %init) -> (i8) {
          %s = scf.for %kw = %c0 to %c3 step %c1 iter_args(%b = %a) -> (i8) {
            %ih = arith.addi %oh, %kh : index
            %iw = arith.addi %ow, %kw : index
            %v = memref.load %in[%c0, %ih, %iw, %c] : memref<1x10x10x4xi8>
            %m = arith.maxsi %b, %v : i8
            scf.yield %m : i8
          }
          scf.yield %s : i8
        }
        memref.store %r, %out[%c0, %oh, %ow, %c] : memref<1x4x4x4xi8>
      }
    }
  }
  return
}

// -----

// A loop that carries nothing is not a window: it runs the length of a row, and
// straightening it out would multiply the code for no chain to shorten.
// `--unroll-elementwise-loops` is what handles those, and it measured two
// iterations to be the right factor.
// CHECK-LABEL: func.func @an_elementwise_row
// CHECK:         scf.for
// CHECK-COUNT-1: memref.load
// CHECK-NOT:     memref.load
func.func @an_elementwise_row(%in: memref<3xf32>, %out: memref<3xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %two = arith.constant 2.000000e+00 : f32
  scf.for %i = %c0 to %c3 step %c1 {
    %v = memref.load %in[%i] : memref<3xf32>
    %m = arith.mulf %v, %two : f32
    memref.store %m, %out[%i] : memref<3xf32>
  }
  return
}

// -----

// A reduction over a whole row is not a window either -- 64 trips is past the
// budget, and the loop stays.
// CHECK-LABEL: func.func @a_long_reduction
// CHECK:         scf.for {{.*}}iter_args
func.func @a_long_reduction(%in: memref<64xf32>, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %init = memref.load %out[%c0] : memref<1xf32>
  %r = scf.for %i = %c0 to %c64 step %c1 iter_args(%a = %init) -> (f32) {
    %v = memref.load %in[%i] : memref<64xf32>
    %s = arith.addf %a, %v : f32
    scf.yield %s : f32
  }
  memref.store %r, %out[%c0] : memref<1xf32>
  return
}

// -----

// The budget is on the **unrolled** body, and it is what keeps a 7x7 window's
// outer level: the inner one straightens to seven loads, and seven copies of
// *that* is past what is worth the instruction cache. One loop stays, around a
// straight run.
// CHECK-LABEL: func.func @a_seven_by_seven_window
// CHECK:         %[[INIT:.*]] = memref.load %arg1
// CHECK:         scf.for {{.*}}iter_args(%{{.*}} = %[[INIT]])
// CHECK-COUNT-7: memref.load %arg0
// CHECK-NOT:     memref.load %arg0
func.func @a_seven_by_seven_window(%in: memref<1x16x16x4xi8>, %out: memref<1x8x8x4xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  scf.for %oh = %c0 to %c8 step %c1 {
    scf.for %ow = %c0 to %c8 step %c1 {
      scf.for %c = %c0 to %c8 step %c1 {
        %init = memref.load %out[%c0, %oh, %ow, %c] : memref<1x8x8x4xi8>
        %r = scf.for %kh = %c0 to %c7 step %c1 iter_args(%a = %init) -> (i8) {
          %s = scf.for %kw = %c0 to %c7 step %c1 iter_args(%b = %a) -> (i8) {
            %ih = arith.addi %oh, %kh : index
            %iw = arith.addi %ow, %kw : index
            %v = memref.load %in[%c0, %ih, %iw, %c] : memref<1x16x16x4xi8>
            %m = arith.maxsi %b, %v : i8
            scf.yield %m : i8
          }
          scf.yield %s : i8
        }
        memref.store %r, %out[%c0, %oh, %ow, %c] : memref<1x8x8x4xi8>
      }
    }
  }
  return
}

// -----

// A call in the loop is an accelerator or a copy: its cost is not the loop, and
// nine copies of it is nine times the code for nothing.
// CHECK-LABEL: func.func @a_call_in_the_window
// CHECK:         scf.for {{.*}}iter_args
// CHECK-COUNT-1: func.call
// CHECK-NOT:     func.call
func.func @a_call_in_the_window(%in: memref<4xf32>, %out: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init = memref.load %out[%c0] : memref<1xf32>
  %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%a = %init) -> (f32) {
    func.call @sink(%in) : (memref<4xf32>) -> ()
    %v = memref.load %in[%i] : memref<4xf32>
    %s = arith.addf %a, %v : f32
    scf.yield %s : f32
  }
  memref.store %r, %out[%c0] : memref<1xf32>
  return
}
func.func private @sink(memref<4xf32>)
