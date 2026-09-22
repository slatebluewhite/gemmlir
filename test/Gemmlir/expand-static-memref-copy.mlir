// RUN: gemmlir-opt --expand-static-memref-copy %s | FileCheck %s

// The runtime's copy reads a descriptor, works out how much of the shape is
// packed on both sides, decides whether the run is word-aligned and then runs
// an odometer -- the same answer every time for a given call site. Written out
// here the run length is a constant, so its loads and stores are unrolled and
// the index arithmetic is two `addi`.

// im2col: `kw` and `c` are packed on both sides, `kh` is not, so the run is 24
// bytes and three axes are left to walk.
// CHECK-LABEL: func.func @im2col
// CHECK:         %[[S:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0], [1], [2], [3], [4, 5]]
// CHECK:         %[[D:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0], [1], [2], [3], [4, 5]]
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               %[[V:.*]] = vector.load %[[S]]
// CHECK-SAME:            {alignment = 8 : i64}
// CHECK-SAME:            vector<24xi8>
// CHECK:               vector.store %[[V]], %[[D]]
// CHECK-SAME:            {alignment = 8 : i64}
// CHECK-NOT:     memref.copy
func.func @im2col() {
  %pad = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xi8>
  %out = memref.alloc() {alignment = 64 : i64} : memref<1x16x16x3x3x8xi8>
  %win = memref.reinterpret_cast %pad to offset: [0], sizes: [1, 16, 16, 3, 3, 8],
         strides: [2592, 144, 8, 144, 8, 1]
         : memref<1x18x18x8xi8> to memref<1x16x16x3x3x8xi8, strided<[2592, 144, 8, 144, 8, 1]>>
  memref.copy %win, %out
    : memref<1x16x16x3x3x8xi8, strided<[2592, 144, 8, 144, 8, 1]>> to memref<1x16x16x3x3x8xi8>
  return
}

// A run of exactly one word is the one size LLVM takes apart -- `vector<8xi8>`
// comes out as eight byte loads and eight byte stores -- so it is left to the
// runtime, which copies it as one `ld`/`sd` pair.
// CHECK-LABEL: func.func @one_word_run
// CHECK:         memref.copy
// CHECK-NOT:     vector.load
func.func @one_word_run() {
  %big = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
  %out = memref.alloc() {alignment = 64 : i64} : memref<16x8xi8>
  %win = memref.subview %big[0, 0] [16, 8] [1, 1]
    : memref<16x16xi8> to memref<16x8xi8, strided<[16, 1]>>
  memref.copy %win, %out : memref<16x8xi8, strided<[16, 1]>> to memref<16x8xi8>
  return
}

// Past 64 bytes the runtime calls `memcpy`, which beats a long unrolled
// sequence.
// CHECK-LABEL: func.func @long_run
// CHECK:         memref.copy
// CHECK-NOT:     vector.load
func.func @long_run() {
  %big = memref.alloc() {alignment = 64 : i64} : memref<16x256xi8>
  %out = memref.alloc() {alignment = 64 : i64} : memref<16x128xi8>
  %win = memref.subview %big[0, 0] [16, 128] [1, 1]
    : memref<16x256xi8> to memref<16x128xi8, strided<[256, 1]>>
  memref.copy %win, %out : memref<16x128xi8, strided<[256, 1]>> to memref<16x128xi8>
  return
}

// A run that is not a whole number of words cannot be moved as aligned
// vectors -- im2col over a three-channel image has runs of nine -- so it moves
// as its elements instead, which needs nothing proved: a load of the element
// type is aligned wherever the element is. Unrolled at constant offsets it is
// nine loads and nine stores with no inner loop, against the runtime copying
// one byte at a time with the length in a register.
// CHECK-LABEL: func.func @ragged_run
// CHECK:         scf.for
// CHECK-COUNT-9:   memref.load
// CHECK-NOT:     memref.copy
// CHECK-NOT:     vector.load
func.func @ragged_run() {
  %big = memref.alloc() {alignment = 64 : i64} : memref<16x32xi8>
  %out = memref.alloc() {alignment = 64 : i64} : memref<16x9xi8>
  %win = memref.subview %big[0, 0] [16, 9] [1, 1]
    : memref<16x32xi8> to memref<16x9xi8, strided<[32, 1]>>
  memref.copy %win, %out : memref<16x9xi8, strided<[32, 1]>> to memref<16x9xi8>
  return
}

// A ragged run needs no alignment, so an opaque base is fine for it.
// CHECK-LABEL: func.func @ragged_unknown_base
// CHECK:         scf.for
// CHECK-COUNT-9:   memref.load
// CHECK-NOT:     memref.copy
func.func @ragged_unknown_base(%big: memref<16x32xi8>, %out: memref<16x9xi8>) {
  %win = memref.subview %big[0, 0] [16, 9] [1, 1]
    : memref<16x32xi8> to memref<16x9xi8, strided<[32, 1]>>
  memref.copy %win, %out : memref<16x9xi8, strided<[32, 1]>> to memref<16x9xi8>
  return
}

// Unrolling stops being the answer once the run is long: thirty elements is
// sixty memory operations written out, and the runtime's byte loop is better.
// The cap was sixteen and is twenty-four -- see the pass, and note that the
// original sixteen was a judgement rather than a measurement.
// CHECK-LABEL: func.func @long_ragged_run
// CHECK:         memref.copy
// CHECK-NOT:     memref.load
func.func @long_ragged_run() {
  %big = memref.alloc() {alignment = 64 : i64} : memref<16x48xi8>
  %out = memref.alloc() {alignment = 64 : i64} : memref<16x30xi8>
  %win = memref.subview %big[0, 0] [16, 30] [1, 1]
    : memref<16x48xi8> to memref<16x30xi8, strided<[48, 1]>>
  memref.copy %win, %out : memref<16x30xi8, strided<[48, 1]>> to memref<16x30xi8>
  return
}

// Packed all the way through is already one `memcpy` in the existing lowering,
// which does it better than a nest of one iteration.
// CHECK-LABEL: func.func @fully_packed
// CHECK:         memref.copy
// CHECK-NOT:     vector.load
func.func @fully_packed() {
  %a = memref.alloc() {alignment = 64 : i64} : memref<16x24xi8>
  %b = memref.alloc() {alignment = 64 : i64} : memref<16x24xi8>
  memref.copy %a, %b : memref<16x24xi8> to memref<16x24xi8>
  return
}

// Nothing here says where a function argument starts, and an unaligned
// `vector.load` is byte accesses again -- slower than the call it replaced.
// CHECK-LABEL: func.func @unknown_alignment
// CHECK:         memref.copy
// CHECK-NOT:     vector.load
func.func @unknown_alignment(%big: memref<16x32xi8>, %out: memref<16x24xi8>) {
  %win = memref.subview %big[0, 0] [16, 24] [1, 1]
    : memref<16x32xi8> to memref<16x24xi8, strided<[32, 1]>>
  memref.copy %win, %out : memref<16x24xi8, strided<[32, 1]>> to memref<16x24xi8>
  return
}

// A stride that is not a multiple of the word size drifts the run off its
// alignment after the first step.
// CHECK-LABEL: func.func @odd_stride
// CHECK:         memref.copy
// CHECK-NOT:     vector.load
func.func @odd_stride() {
  %big = memref.alloc() {alignment = 64 : i64} : memref<16x28xi8>
  %out = memref.alloc() {alignment = 64 : i64} : memref<16x24xi8>
  %win = memref.subview %big[0, 0] [16, 24] [1, 1]
    : memref<16x28xi8> to memref<16x24xi8, strided<[28, 1]>>
  memref.copy %win, %out : memref<16x24xi8, strided<[28, 1]>> to memref<16x24xi8>
  return
}

// f32 is the same question with four bytes an element: four of them is a run
// of sixteen.
// CHECK-LABEL: func.func @floats
// CHECK:         vector.load
// CHECK-SAME:      vector<4xf32>
func.func @floats() {
  %big = memref.alloc() {alignment = 64 : i64} : memref<8x16xf32>
  %out = memref.alloc() {alignment = 64 : i64} : memref<8x4xf32>
  %win = memref.subview %big[0, 0] [8, 4] [1, 1]
    : memref<8x16xf32> to memref<8x4xf32, strided<[16, 1]>>
  memref.copy %win, %out : memref<8x4xf32, strided<[16, 1]>> to memref<8x4xf32>
  return
}


// -----

// A ragged run of 21 bytes: DenseNet's stem packs im2col over a three-channel
// image with a 7x7 kernel, so the packed suffix is 7*3. It is not a whole
// number of words, so it moves as elements -- 21 loads and 21 stores at
// constant offsets, against a runtime call for each of 7168 runs.

// CHECK-LABEL: func.func @ragged_21
// CHECK-NOT:     memref.copy
// CHECK:         scf.for
// CHECK-COUNT-21: memref.load
func.func @ragged_21(%src: memref<1x32x32x7x7x3xi8, strided<[14700, 420, 6, 210, 3, 1]>>,
                     %dst: memref<1x32x32x7x7x3xi8>) {
  memref.copy %src, %dst
    : memref<1x32x32x7x7x3xi8, strided<[14700, 420, 6, 210, 3, 1]>>
      to memref<1x32x32x7x7x3xi8>
  return
}
