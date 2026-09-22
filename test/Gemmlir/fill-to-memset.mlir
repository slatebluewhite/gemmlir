// A convolution's padding bufferizes into a fill of the whole padded buffer
// plus a copy of the real input into the middle of it, and the fill is one
// scalar store per element. A repeated byte is a memset instead.
//
// It runs after --convert-linalg-to-gemmlir on purpose: a zero fill is how an
// accumulator is proved to start at zero, and the conversion reads them.

// RUN: gemmlir-opt --fill-to-memset %s | FileCheck %s

// CHECK-LABEL: func.func @zero_fill
// CHECK:         gemmlir.memset(%arg0) {value = 0 : i8} : memref<1x18x18x8xi8>
// CHECK-NOT:     linalg.fill
func.func @zero_fill(%b: memref<1x18x18x8xi8>) {
  %z = arith.constant 0 : i8
  linalg.fill ins(%z : i8) outs(%b : memref<1x18x18x8xi8>)
  return
}

// Zero is zero whatever the type: an f32 zero is four zero bytes.
// CHECK-LABEL: func.func @zero_f32
// CHECK:         gemmlir.memset(%arg0) {value = 0 : i8} : memref<4x9xf32>
func.func @zero_f32(%b: memref<4x9xf32>) {
  %z = arith.constant 0.0 : f32
  linalg.fill ins(%z : f32) outs(%b : memref<4x9xf32>)
  return
}

// -1 as i32 is four 0xff bytes, which a memset can write.
// CHECK-LABEL: func.func @all_ones
// CHECK:         gemmlir.memset(%arg0) {value = -1 : i8} : memref<4x9xi32>
func.func @all_ones(%b: memref<4x9xi32>) {
  %m = arith.constant -1 : i32
  linalg.fill ins(%m : i32) outs(%b : memref<4x9xi32>)
  return
}

// 1.0f is 00 00 80 3F -- four different bytes, and there is no memset for it.
// CHECK-LABEL: func.func @one_f32
// CHECK:         linalg.fill
// CHECK-NOT:     gemmlir.memset
func.func @one_f32(%b: memref<4x9xf32>) {
  %o = arith.constant 1.0 : f32
  linalg.fill ins(%o : f32) outs(%b : memref<4x9xf32>)
  return
}

// A window of a bigger buffer is not contiguous, so one memset would spill into
// what sits between its rows.
// CHECK-LABEL: func.func @strided_window
// CHECK:         linalg.fill
// CHECK-NOT:     gemmlir.memset
func.func @strided_window(%b: memref<4x9xi8, strided<[16, 1], offset: 3>>) {
  %z = arith.constant 0 : i8
  linalg.fill ins(%z : i8) outs(%b : memref<4x9xi8, strided<[16, 1], offset: 3>>)
  return
}

// An offset alone is fine: the rows still follow one another.
// CHECK-LABEL: func.func @offset_is_fine
// CHECK:         gemmlir.memset
func.func @offset_is_fine(%b: memref<4x9xi8, strided<[9, 1], offset: 12>>) {
  %z = arith.constant 0 : i8
  linalg.fill ins(%z : i8) outs(%b : memref<4x9xi8, strided<[9, 1], offset: 12>>)
  return
}

// Bufferizing a tensor.pad leaves the fill as a linalg.map with no inputs whose
// body yields the constant, not as a linalg.fill.
// CHECK-LABEL: func.func @pad_fill_is_a_map
// CHECK:         gemmlir.memset(%arg0) {value = 0 : i8} : memref<1x18x18x8xi8>
// CHECK-NOT:     linalg.map
func.func @pad_fill_is_a_map(%b: memref<1x18x18x8xi8>) {
  %z = arith.constant 0 : i8
  linalg.map outs(%b : memref<1x18x18x8xi8>)
    (%init: i8) {
      linalg.yield %z : i8
    }
  return
}

// The same shape but yielding what was already there is a no-op, not a fill.
// CHECK-LABEL: func.func @map_yields_its_init
// CHECK:         linalg.map
// CHECK-NOT:     gemmlir.memset
func.func @map_yields_its_init(%b: memref<1x18x18x8xi8>) {
  linalg.map outs(%b : memref<1x18x18x8xi8>)
    (%init: i8) {
      linalg.yield %init : i8
    }
  return
}

// A reduction's accumulator initialisation stays a loop. `gemmlir_memset` is an
// opaque call, so the loop that follows would have to treat the buffer as
// unknown-modified and reload the accumulator on every step instead of keeping
// it in a register. Measured on the board, converting these took `shub` from
// 56.6 to 83.7 ms and `apb` from 42.8 to 59.0, while every model without a
// convolution left as a scalar loop got faster.
// CHECK-LABEL: func.func @accumulator_stays_a_loop
// CHECK:         linalg.fill
// CHECK-NOT:     gemmlir.memset
func.func @accumulator_stays_a_loop(%in: memref<1x18x18x8xi8>, %f: memref<3x3x8x8xi8>,
                                    %acc: memref<1x16x16x8xi32>) {
  %z = arith.constant 0 : i32
  linalg.fill ins(%z : i32) outs(%acc : memref<1x16x16x8xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %f : memref<1x18x18x8xi8>, memref<3x3x8x8xi8>) outs(%acc : memref<1x16x16x8xi32>)
  return
}

// A padding fill is not that: what follows writes a *window* of the buffer, not
// the whole of it, and reads nothing back.
// CHECK-LABEL: func.func @padding_fill_converts
// CHECK:         gemmlir.memset
func.func @padding_fill_converts(%real: memref<1x16x16x8xi8>, %padded: memref<1x18x18x8xi8>) {
  %z = arith.constant 0 : i8
  linalg.fill ins(%z : i8) outs(%padded : memref<1x18x18x8xi8>)
  %win = memref.subview %padded[0, 1, 1, 0] [1, 16, 16, 8] [1, 1, 1, 1]
    : memref<1x18x18x8xi8> to memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>
  memref.copy %real, %win : memref<1x16x16x8xi8> to memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>
  return
}

// Nor in front of a reduction that *reads* it. A convolution left as a scalar
// loop reading a padded buffer loses the same way its accumulator would: the
// opaque call is an alias barrier across the loop. This is `shub`'s shape, and
// converting this one fill cost it 56.6 -> 59.5 ms.
// CHECK-LABEL: func.func @read_by_a_reduction
// CHECK:         linalg.fill
// CHECK-NOT:     gemmlir.memset
func.func @read_by_a_reduction(%real: memref<1x16x16x8xi8>, %padded: memref<1x18x18x8xi8>,
                               %f: memref<3x3x8x8xi8>, %acc: memref<1x16x16x8xi32>) {
  %z = arith.constant 0 : i8
  linalg.fill ins(%z : i8) outs(%padded : memref<1x18x18x8xi8>)
  %win = memref.subview %padded[0, 1, 1, 0] [1, 16, 16, 8] [1, 1, 1, 1]
    : memref<1x18x18x8xi8> to memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>
  memref.copy %real, %win : memref<1x16x16x8xi8> to memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%padded, %f : memref<1x18x18x8xi8>, memref<3x3x8x8xi8>) outs(%acc : memref<1x16x16x8xi32>)
  return
}

// An elementwise loop over the buffer is not a reason to refuse: it has no
// accumulator to spill.
// CHECK-LABEL: func.func @read_by_an_elementwise_loop
// CHECK:         gemmlir.memset
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @read_by_an_elementwise_loop(%padded: memref<1x18x18x8xi8>,
                                       %out: memref<1x18x18x8xf32>) {
  %z = arith.constant 0 : i8
  %s = arith.constant 2.000000e-02 : f32
  linalg.fill ins(%z : i8) outs(%padded : memref<1x18x18x8xi8>)
  linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%padded : memref<1x18x18x8xi8>) outs(%out : memref<1x18x18x8xf32>) {
  ^bb0(%v: i8, %o: f32):
    %e = arith.extsi %v : i8 to i32
    %g = arith.sitofp %e : i32 to f32
    %m = arith.mulf %g, %s : f32
    linalg.yield %m : f32
  }
  return
}

// Writing into a filled buffer without reading it back is not an accumulator. A
// convolution's padding is filled and then the real input is written into the
// middle of it, and that write is elementwise with no accumulator to spill --
// treating it as one kept every padding in the grouped family a scalar store
// loop.
// CHECK-LABEL: func.func @written_into_but_not_read
// CHECK:         gemmlir.memset
func.func @written_into_but_not_read(%real: memref<1x16x16x8xf32>, %padded: memref<1x18x18x8xi8>) {
  %z = arith.constant 0 : i8
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.fill ins(%z : i8) outs(%padded : memref<1x18x18x8xi8>)
  %win = memref.subview %padded[0, 1, 1, 0] [1, 16, 16, 8] [1, 1, 1, 1]
    : memref<1x18x18x8xi8> to memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>
  linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%real : memref<1x16x16x8xf32>) outs(%win : memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>) {
  ^bb0(%v: f32, %o: i8):
    %q = arith.divf %v, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// Reading it back while writing is an accumulator even without a reduction.
// CHECK-LABEL: func.func @read_back_while_writing
// CHECK:         linalg.fill
// CHECK-NOT:     gemmlir.memset
func.func @read_back_while_writing(%x: memref<1x16x16x8xf32>, %acc: memref<1x16x16x8xf32>) {
  %z = arith.constant 0.0 : f32
  linalg.fill ins(%z : f32) outs(%acc : memref<1x16x16x8xf32>)
  linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%x : memref<1x16x16x8xf32>) outs(%acc : memref<1x16x16x8xf32>) {
  ^bb0(%v: f32, %o: f32):
    %a = arith.addf %v, %o : f32
    linalg.yield %a : f32
  }
  return
}

// -----

// `--plan-static-buffers` puts every tensor at its own offset of one global
// arena. Walking a `memref.view` through to that global made every buffer in
// the function share a base, so the two guards -- "is this buffer touched by a
// later loop" -- answered yes for everything and this pass converted almost
// nothing. The walk stops at the view.

// CHECK-LABEL: func @two_views_of_one_arena
// CHECK:         gemmlir.memset
// CHECK-SAME:      {value = 0 : i8}
#id = affine_map<(d0) -> (d0)>
memref.global "private" @arena : memref<4096xi8>
func.func @two_views_of_one_arena() {
  %c0 = arith.constant 0 : index
  %c2048 = arith.constant 2048 : index
  %z = arith.constant 0 : i8
  %g = memref.get_global @arena : memref<4096xi8>
  %a = memref.view %g[%c0][] : memref<4096xi8> to memref<64xi8>
  %b = memref.view %g[%c2048][] : memref<4096xi8> to memref<64xi8>
  %acc = memref.alloc() : memref<i8>
  linalg.fill ins(%z : i8) outs(%a : memref<64xi8>)
  // A reduction over a *different* view of the same arena: not this buffer.
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
                  iterator_types = ["reduction"]}
    ins(%b : memref<64xi8>) outs(%acc : memref<i8>) {
  ^bb0(%in: i8, %out: i8):
    %m = arith.maxsi %in, %out : i8
    linalg.yield %m : i8
  }
  return
}

// -----

// A slab whose outermost extent is one carries the whole buffer's stride there,
// and it is still a single contiguous run -- a dimension of size one never
// steps. This is the shape `--fill-only-the-border` leaves behind for the last
// rows of a padding.

// CHECK-LABEL: func @a_size_one_outer_dimension
// CHECK:         gemmlir.memset
memref.global "private" @arena2 : memref<400000xi8>
func.func @a_size_one_outer_dimension() {
  %c0 = arith.constant 0 : index
  %z = arith.constant 0 : i8
  %g = memref.get_global @arena2 : memref<400000xi8>
  %v = memref.view %g[%c0][] : memref<400000xi8> to memref<1x50x50x64xi8>
  %s = memref.subview %v[0, 48, 0, 0] [1, 2, 50, 64] [1, 1, 1, 1]
    : memref<1x50x50x64xi8> to memref<1x2x50x64xi8, strided<[160000, 3200, 64, 1], offset: 153600>>
  linalg.fill ins(%z : i8) outs(%s : memref<1x2x50x64xi8, strided<[160000, 3200, 64, 1], offset: 153600>>)
  return
}

// -----

// The last *columns* of a padding are a run per row: one `memset` per run,
// under a loop over the rows. 48 runs of 128 bytes here, where refusing left
// 6,144 scalar stores.

// CHECK-LABEL: func @a_run_per_row
// CHECK:         scf.for %[[R:.*]] = %{{.*}} to %{{.*}} step
// CHECK:           %[[S:.*]] = memref.subview %{{.*}}[0, %[[R]], 0, 0] [1, 1, 2, 64]
// CHECK:           gemmlir.memset(%[[S]])
// CHECK-NOT:     linalg.fill
memref.global "private" @arena3 : memref<400000xi8>
func.func @a_run_per_row() {
  %c0 = arith.constant 0 : index
  %z = arith.constant 0 : i8
  %g = memref.get_global @arena3 : memref<400000xi8>
  %v = memref.view %g[%c0][] : memref<400000xi8> to memref<1x50x50x64xi8>
  %s = memref.subview %v[0, 0, 48, 0] [1, 48, 2, 64] [1, 1, 1, 1]
    : memref<1x50x50x64xi8> to memref<1x48x2x64xi8, strided<[160000, 3200, 64, 1], offset: 3072>>
  linalg.fill ins(%z : i8) outs(%s : memref<1x48x2x64xi8, strided<[160000, 3200, 64, 1], offset: 3072>>)
  return
}

// -----

// `--fill-to-memset=below-reduction=1` also converts a fill whose buffer a
// later reduction *reads*. The default refuses it: `gemmlir_memset` is an
// opaque call, and when the reduction still carried its accumulator through
// memory a call in front of one cost more than the fill saved.
// RUN: gemmlir-opt --fill-to-memset="below-reduction=1" %s --split-input-file \
// RUN:   | FileCheck %s --check-prefix=BELOW

// CHECK-LABEL: func @read_by_a_pool
// CHECK:         linalg.fill
// CHECK-NOT:     gemmlir.memset
// BELOW-LABEL: func @read_by_a_pool
// BELOW:         gemmlir.memset
func.func @read_by_a_pool(%win: memref<2x2xi8>, %out: memref<1x4x4x64xi8>) {
  %c0 = arith.constant 0 : index
  %lo = arith.constant -128 : i8
  %pad = memref.alloc() : memref<1x8x8x64xi8>
  linalg.fill ins(%lo : i8) outs(%pad : memref<1x8x8x64xi8>)
  linalg.pooling_nhwc_max {strides = dense<2> : tensor<2xi64>,
                           dilations = dense<1> : tensor<2xi64>}
    ins(%pad, %win : memref<1x8x8x64xi8>, memref<2x2xi8>)
    outs(%out : memref<1x4x4x64xi8>)
  return
}

// -----

// A run shorter than a cache line does not pay for a call: eight bytes a row
// is 4,096 calls where the stores were 32,768.

// CHECK-LABEL: func @a_run_too_short
// CHECK:         linalg.fill
// CHECK-NOT:     gemmlir.memset
memref.global "private" @arena9 : memref<400000xi8>
func.func @a_run_too_short() {
  %c0 = arith.constant 0 : index
  %z = arith.constant 0 : i8
  %g = memref.get_global @arena9 : memref<400000xi8>
  %v = memref.view %g[%c0][] : memref<400000xi8> to memref<1x64x64x64xi8>
  %s = memref.subview %v[0, 0, 0, 0] [1, 64, 64, 8] [1, 1, 1, 1]
    : memref<1x64x64x64xi8> to memref<1x64x64x8xi8, strided<[262144, 4096, 64, 1]>>
  linalg.fill ins(%z : i8) outs(%s : memref<1x64x64x8xi8, strided<[262144, 4096, 64, 1]>>)
  return
}

// -----

// Two dimensions step over the gaps, so there are two loops.

// CHECK-LABEL: func @two_loops
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             gemmlir.memset
// CHECK-NOT:     linalg.fill
memref.global "private" @arena10 : memref<4000000xi8>
func.func @two_loops() {
  %c0 = arith.constant 0 : index
  %z = arith.constant 0 : i8
  %g = memref.get_global @arena10 : memref<4000000xi8>
  %v = memref.view %g[%c0][] : memref<4000000xi8> to memref<4x50x50x64xi8>
  %s = memref.subview %v[0, 0, 0, 0] [4, 50, 2, 64] [1, 1, 1, 1]
    : memref<4x50x50x64xi8> to memref<4x50x2x64xi8, strided<[160000, 3200, 64, 1]>>
  linalg.fill ins(%z : i8) outs(%s : memref<4x50x2x64xi8, strided<[160000, 3200, 64, 1]>>)
  return
}

// -----

// A strided fill that a later loop accumulates into is still that loop's
// accumulator, and the run form does not change that.

// CHECK-LABEL: func @strided_accumulator
// CHECK:         linalg.fill
// CHECK-NOT:     gemmlir.memset
func.func @strided_accumulator(%win: memref<2x2xi8>, %out: memref<1x4x4x64xi8>) {
  %lo = arith.constant -128 : i8
  %big = memref.alloc() : memref<1x8x8x128xi8>
  %s = memref.subview %big[0, 0, 0, 0] [1, 8, 8, 64] [1, 1, 1, 1]
    : memref<1x8x8x128xi8> to memref<1x8x8x64xi8, strided<[8192, 1024, 128, 1]>>
  linalg.fill ins(%lo : i8) outs(%s : memref<1x8x8x64xi8, strided<[8192, 1024, 128, 1]>>)
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%s : memref<1x8x8x64xi8, strided<[8192, 1024, 128, 1]>>)
    outs(%s : memref<1x8x8x64xi8, strided<[8192, 1024, 128, 1]>>) {
  ^bb0(%in: i8, %o: i8):
    %m = arith.maxsi %in, %o : i8
    linalg.yield %m : i8
  }
  return
}
