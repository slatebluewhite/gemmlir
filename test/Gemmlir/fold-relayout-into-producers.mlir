// RUN: gemmlir-opt --fold-relayout-into-producers --split-input-file %s | FileCheck %s

// A grouped convolution's tails each dequantize their own channels into an
// NHWC buffer, which is then relaid out to the NCHW the model returns. Each
// tail already walks its own iteration space, so writing the permuted slice
// costs it nothing and the relayout goes away.

#nhwc = affine_map<(n, h, w, c) -> (n, h, w, c)>
#nchw = affine_map<(n, h, w, c) -> (n, c, h, w)>

// The permutation ends up in the tail's own output map.
// CHECK-DAG:   #[[NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: func.func @two_tails
// CHECK:         %[[OUT:.*]] = memref.alloc() : memref<1x16x8x8xf32>
// CHECK:         %[[A:.*]] = memref.subview %[[OUT]][0, 0, 0, 0] [1, 8, 8, 8] [1, 1, 1, 1]
// CHECK:         linalg.generic {indexing_maps = [#{{.*}}, #[[NCHW]]]
// CHECK-SAME:      outs(%[[A]]
// CHECK:         %[[B:.*]] = memref.subview %[[OUT]][0, 8, 0, 0] [1, 8, 8, 8] [1, 1, 1, 1]
// CHECK:         linalg.generic {indexing_maps = [#{{.*}}, #[[NCHW]]]
// CHECK-SAME:      outs(%[[B]]
// CHECK-NOT:     linalg.transpose
// CHECK:         return %[[OUT]]
func.func @two_tails(%a: memref<1x8x8x8xi32>, %b: memref<1x8x8x8xi32>)
    -> memref<1x16x8x8xf32> {
  %s = arith.constant 0.013 : f32
  %nhwc = memref.alloc() : memref<1x8x8x16xf32>
  %lo = memref.subview %nhwc[0, 0, 0, 0] [1, 8, 8, 8] [1, 1, 1, 1]
    : memref<1x8x8x16xf32> to memref<1x8x8x8xf32, strided<[1024, 128, 16, 1]>>
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%a : memref<1x8x8x8xi32>) outs(%lo : memref<1x8x8x8xf32, strided<[1024, 128, 16, 1]>>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  }
  %hi = memref.subview %nhwc[0, 0, 0, 8] [1, 8, 8, 8] [1, 1, 1, 1]
    : memref<1x8x8x16xf32> to memref<1x8x8x8xf32, strided<[1024, 128, 16, 1], offset: 8>>
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%b : memref<1x8x8x8xi32>) outs(%hi : memref<1x8x8x8xf32, strided<[1024, 128, 16, 1], offset: 8>>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  }
  %nchw = memref.alloc() : memref<1x16x8x8xf32>
  linalg.transpose ins(%nhwc : memref<1x8x8x16xf32>) outs(%nchw : memref<1x16x8x8xf32>) permutation = [0, 3, 1, 2]
  memref.dealloc %nhwc : memref<1x8x8x16xf32>
  return %nchw : memref<1x16x8x8xf32>
}

// Slices that leave a gap would leave part of the relayout's result unwritten.
// CHECK-LABEL: func.func @does_not_cover
// CHECK:         linalg.transpose
func.func @does_not_cover(%a: memref<1x8x8x8xi32>) -> memref<1x16x8x8xf32> {
  %s = arith.constant 0.013 : f32
  %nhwc = memref.alloc() : memref<1x8x8x16xf32>
  %lo = memref.subview %nhwc[0, 0, 0, 0] [1, 8, 8, 8] [1, 1, 1, 1]
    : memref<1x8x8x16xf32> to memref<1x8x8x8xf32, strided<[1024, 128, 16, 1]>>
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%a : memref<1x8x8x8xi32>) outs(%lo : memref<1x8x8x8xf32, strided<[1024, 128, 16, 1]>>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  }
  %nchw = memref.alloc() : memref<1x16x8x8xf32>
  linalg.transpose ins(%nhwc : memref<1x8x8x16xf32>) outs(%nchw : memref<1x16x8x8xf32>) permutation = [0, 3, 1, 2]
  memref.dealloc %nhwc : memref<1x8x8x16xf32>
  return %nchw : memref<1x16x8x8xf32>
}

// Something else reading the buffer means the writes cannot simply move.
// CHECK-LABEL: func.func @read_as_well
// CHECK:         linalg.transpose
func.func @read_as_well(%a: memref<1x8x8x16xi32>, %other: memref<1x8x8x16xf32>)
    -> memref<1x16x8x8xf32> {
  %s = arith.constant 0.013 : f32
  %nhwc = memref.alloc() : memref<1x8x8x16xf32>
  %all = memref.subview %nhwc[0, 0, 0, 0] [1, 8, 8, 16] [1, 1, 1, 1]
    : memref<1x8x8x16xf32> to memref<1x8x8x16xf32, strided<[1024, 128, 16, 1]>>
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%a : memref<1x8x8x16xi32>) outs(%all : memref<1x8x8x16xf32, strided<[1024, 128, 16, 1]>>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  }
  memref.copy %nhwc, %other : memref<1x8x8x16xf32> to memref<1x8x8x16xf32>
  %nchw = memref.alloc() : memref<1x16x8x8xf32>
  linalg.transpose ins(%nhwc : memref<1x8x8x16xf32>) outs(%nchw : memref<1x16x8x8xf32>) permutation = [0, 3, 1, 2]
  memref.dealloc %nhwc : memref<1x8x8x16xf32>
  return %nchw : memref<1x16x8x8xf32>
}

// A relayout of something the function was handed is not a temporary this pass
// can account for.
// CHECK-LABEL: func.func @not_a_temporary
// CHECK:         linalg.transpose
func.func @not_a_temporary(%nhwc: memref<1x8x8x16xf32>) -> memref<1x16x8x8xf32> {
  %nchw = memref.alloc() : memref<1x16x8x8xf32>
  linalg.transpose ins(%nhwc : memref<1x8x8x16xf32>) outs(%nchw : memref<1x16x8x8xf32>) permutation = [0, 3, 1, 2]
  return %nchw : memref<1x16x8x8xf32>
}

// -----

// **The slices do not have to be the same width.** A grouped convolution's four
// tails are, and requiring it was harmless there; an Inception block's four
// branches are 128, 192, 96 and 64 channels, and the rule refused both of
// GoogLeNet's relayouts -- `1x480x12x12` and `1x832x6x6`, together **17.7% of
// the model** in a pure `flw`/`fsw` copy, by program-counter sampling.
//
// What the rewrite actually needs is that the slices tile one axis and cover
// it: sorted by offset they start at zero, meet exactly, and finish at the end.
// CHECK-LABEL: func.func @unequal_branches
// CHECK-NOT:     linalg.transpose
// CHECK:         memref.subview %[[D:.*]][0, 0, 0, 0] [1, 4, 4, 3]
// CHECK:         linalg.generic
// CHECK:         memref.subview %[[D]][0, 0, 0, 3] [1, 4, 4, 5]
// CHECK:         linalg.generic
#nhwc4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @unequal_branches(%a: memref<1x4x4x3xi32>, %b: memref<1x4x4x5xi32>)
    -> memref<1x4x4x8xf32> {
  %s = arith.constant 0.013 : f32
  %nchw = memref.alloc() : memref<1x8x4x4xf32>
  %lo = memref.subview %nchw[0, 0, 0, 0] [1, 3, 4, 4] [1, 1, 1, 1]
    : memref<1x8x4x4xf32> to memref<1x3x4x4xf32, strided<[128, 16, 4, 1]>>
  linalg.generic {indexing_maps = [#nhwc4, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%a : memref<1x4x4x3xi32>) outs(%lo : memref<1x3x4x4xf32, strided<[128, 16, 4, 1]>>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  }
  %hi = memref.subview %nchw[0, 3, 0, 0] [1, 5, 4, 4] [1, 1, 1, 1]
    : memref<1x8x4x4xf32> to memref<1x5x4x4xf32, strided<[128, 16, 4, 1], offset: 48>>
  linalg.generic {indexing_maps = [#nhwc4, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%b : memref<1x4x4x5xi32>) outs(%hi : memref<1x5x4x4xf32, strided<[128, 16, 4, 1], offset: 48>>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  }
  %nhwc = memref.alloc() : memref<1x4x4x8xf32>
  linalg.transpose ins(%nchw : memref<1x8x4x4xf32>) outs(%nhwc : memref<1x4x4x8xf32>) permutation = [0, 2, 3, 1]
  memref.dealloc %nchw : memref<1x8x4x4xf32>
  return %nhwc : memref<1x4x4x8xf32>
}

// -----

// Slices that leave a gap would leave part of the relayout's result unwritten,
// so they are refused however wide they are.
// CHECK-LABEL: func.func @a_gap_is_refused
// CHECK:         linalg.transpose
#nhwc5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @a_gap_is_refused(%a: memref<1x4x4x3xi32>) -> memref<1x4x4x8xf32> {
  %s = arith.constant 0.013 : f32
  %nchw = memref.alloc() : memref<1x8x4x4xf32>
  %lo = memref.subview %nchw[0, 0, 0, 0] [1, 3, 4, 4] [1, 1, 1, 1]
    : memref<1x8x4x4xf32> to memref<1x3x4x4xf32, strided<[128, 16, 4, 1]>>
  linalg.generic {indexing_maps = [#nhwc5, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%a : memref<1x4x4x3xi32>) outs(%lo : memref<1x3x4x4xf32, strided<[128, 16, 4, 1]>>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  }
  %nhwc = memref.alloc() : memref<1x4x4x8xf32>
  linalg.transpose ins(%nchw : memref<1x8x4x4xf32>) outs(%nhwc : memref<1x4x4x8xf32>) permutation = [0, 2, 3, 1]
  memref.dealloc %nchw : memref<1x8x4x4xf32>
  return %nhwc : memref<1x4x4x8xf32>
}

// -----

// One producer writing the buffer entire, which is the degenerate case of "the
// slices tile one axis and cover it". DenseNet's stem is this: the
// accelerator's i32 output is dequantized into an NCHW buffer and the very next
// operation transposes it back to NHWC, so the two permutations cancel and the
// NCHW buffer stops existing.

// CHECK-DAG: #[[IN:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d2, d3, d1)>
// CHECK-DAG: #[[OUT:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @whole_buffer
// CHECK-NOT:     memref<1x64x32x32xf32>
// CHECK-NOT:     linalg.transpose
// CHECK:         linalg.generic {indexing_maps = [#[[IN]], #{{.*}}, #[[OUT]]]
// CHECK-SAME:      outs(%{{.*}} : memref<1x32x32x64xf32>)
#in = affine_map<(d0, d1, d2, d3) -> (0, d2, d3, d1)>
#ch = affine_map<(d0, d1, d2, d3) -> (d1)>
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @whole_buffer(%acc: memref<1x32x32x64xi32>, %bias: memref<64xf32>,
                        %out: memref<1x32x32x64xf32>) {
  %s = arith.constant 4.0 : f32
  %nchw = memref.alloc() {alignment = 64 : i64} : memref<1x64x32x32xf32>
  %nhwc = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x64xf32>
  linalg.generic {indexing_maps = [#in, #ch, #id],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%acc, %bias : memref<1x32x32x64xi32>, memref<64xf32>)
    outs(%nchw : memref<1x64x32x32xf32>) {
  ^bb0(%a: i32, %b: f32, %o: f32):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %p = arith.addf %m, %b : f32
    linalg.yield %p : f32
  }
  linalg.transpose ins(%nchw : memref<1x64x32x32xf32>)
    outs(%nhwc : memref<1x32x32x64xf32>) permutation = [0, 2, 3, 1]
  memref.copy %nhwc, %out : memref<1x32x32x64xf32> to memref<1x32x32x64xf32>
  memref.dealloc %nchw : memref<1x64x32x32xf32>
  memref.dealloc %nhwc : memref<1x32x32x64xf32>
  return
}

// -----

// The producer reads the buffer it writes, so it is an accumulator and moving
// where it writes would change what it reads.

// CHECK-LABEL: func.func @producer_reads_it
// CHECK:         linalg.transpose
#id2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @producer_reads_it(%out: memref<1x32x32x64xf32>) {
  %nchw = memref.alloc() {alignment = 64 : i64} : memref<1x64x32x32xf32>
  %nhwc = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x64xf32>
  linalg.generic {indexing_maps = [#id2, #id2],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%nchw : memref<1x64x32x32xf32>)
    outs(%nchw : memref<1x64x32x32xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  linalg.transpose ins(%nchw : memref<1x64x32x32xf32>)
    outs(%nhwc : memref<1x32x32x64xf32>) permutation = [0, 2, 3, 1]
  memref.copy %nhwc, %out : memref<1x32x32x64xf32> to memref<1x32x32x64xf32>
  memref.dealloc %nchw : memref<1x64x32x32xf32>
  memref.dealloc %nhwc : memref<1x32x32x64xf32>
  return
}
