// `out[i] = in[map(i)]` with a `map` that is linear in the iteration indices is
// a copy between two strided views of the same shape. Written that way the
// runtime copies the longest run that is packed on both sides with one memcpy,
// instead of the load-and-store per element --convert-linalg-to-loops emits.

// RUN: gemmlir-opt --gather-to-memref-copy %s | FileCheck %s

#pack = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
#id6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#tp2 = affine_map<(d0, d1) -> (d1, d0)>
#gather2 = affine_map<(d0, d1) -> (d0 floordiv 3, d1 mod 4)>

// im2col is the one that matters. The source is 1x18x18x8, strides
// (2592, 144, 8, 1); `oh` and `kh` both walk the row axis, `ow` and `kw` both
// walk the column axis, so the view repeats those strides. The destination is
// packed, so the runtime's run is `kw` and `c` together: 24 bytes.
// CHECK-LABEL: func.func @im2col
// CHECK:         %[[V:.*]] = memref.reinterpret_cast %arg0
// CHECK-SAME:      offset: [0]
// CHECK-SAME:      sizes: [1, 16, 16, 3, 3, 8]
// CHECK-SAME:      strides: [2592, 144, 8, 144, 8, 1]
// CHECK:         memref.copy %[[V]], %arg1
// CHECK-NOT:     linalg.generic
func.func @im2col(%in: memref<1x18x18x8xi8>, %out: memref<1x16x16x3x3x8xi8>) {
  linalg.generic {indexing_maps = [#pack, #id6],
                  iterator_types = ["parallel","parallel","parallel","parallel","parallel","parallel"]}
    ins(%in : memref<1x18x18x8xi8>) outs(%out : memref<1x16x16x3x3x8xi8>) {
  ^bb0(%v: i8, %o: i8):
    linalg.yield %v : i8
  }
  return
}

// A stride is still linear: the coefficients change, the shape of the answer
// does not. Here the window steps by two, so `oh` walks 2 rows while `kh` still
// walks one.
// CHECK-LABEL: func.func @im2col_strided
// CHECK:         memref.reinterpret_cast %arg0
// CHECK-SAME:      strides: [2592, 288, 16, 144, 8, 1]
#packs = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d3, d2 * 2 + d4, d5)>
func.func @im2col_strided(%in: memref<1x18x18x8xi8>, %out: memref<1x8x8x3x3x8xi8>) {
  linalg.generic {indexing_maps = [#packs, #id6],
                  iterator_types = ["parallel","parallel","parallel","parallel","parallel","parallel"]}
    ins(%in : memref<1x18x18x8xi8>) outs(%out : memref<1x8x8x3x3x8xi8>) {
  ^bb0(%v: i8, %o: i8):
    linalg.yield %v : i8
  }
  return
}

// A transpose is linear too, but nothing is packed on both sides of it: the run
// is one element, so `memrefCopy` would walk it an element at a time and the
// loop nest is better. Measured on the board, converting it took an attention
// block from 2.47 to 3.03 ms.
// CHECK-LABEL: func.func @transpose
// CHECK:         linalg.generic
// CHECK-NOT:     memref.copy
func.func @transpose(%in: memref<16x32xi8>, %out: memref<32x16xi8>) {
  linalg.generic {indexing_maps = [#tp2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%in : memref<16x32xi8>) outs(%out : memref<32x16xi8>) {
  ^bb0(%v: i8, %o: i8):
    linalg.yield %v : i8
  }
  return
}

// The source's own offset goes into the view's, so a window of a bigger buffer
// reads from the right place. This one is packed throughout, so the whole copy
// is a single `memcpy`.
// CHECK-LABEL: func.func @from_a_subview
// CHECK:         memref.reinterpret_cast %arg0
// CHECK-SAME:      offset: [8]
// CHECK-SAME:      strides: [32, 1]
func.func @from_a_subview(%in: memref<16x32xi8, strided<[32, 1], offset: 8>>,
                          %out: memref<16x32xi8>) {
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%in : memref<16x32xi8, strided<[32, 1], offset: 8>>) outs(%out : memref<16x32xi8>) {
  ^bb0(%v: i8, %o: i8):
    linalg.yield %v : i8
  }
  return
}

// A floordiv or a mod is not a strided view: no coefficient describes it, and
// the loop nest stays.
// CHECK-LABEL: func.func @not_linear
// CHECK:         linalg.generic
// CHECK-NOT:     memref.copy
func.func @not_linear(%in: memref<16x32xi8>, %out: memref<12x20xi8>) {
  linalg.generic {indexing_maps = [#gather2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%in : memref<16x32xi8>) outs(%out : memref<12x20xi8>) {
  ^bb0(%v: i8, %o: i8):
    linalg.yield %v : i8
  }
  return
}

// Not a copy: the body does arithmetic, so it is not a move of bytes.
// CHECK-LABEL: func.func @not_a_copy
// CHECK:         linalg.generic
// CHECK-NOT:     memref.copy
func.func @not_a_copy(%in: memref<16x32xi8>, %out: memref<16x32xi8>) {
  %one = arith.constant 1 : i8
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%in : memref<16x32xi8>) outs(%out : memref<16x32xi8>) {
  ^bb0(%v: i8, %o: i8):
    %s = arith.addi %v, %one : i8
    linalg.yield %s : i8
  }
  return
}

// Overlapping windows of one buffer are refused. `--plan-static-buffers` puts
// every buffer of a function in one arena, so "different allocations" cannot be
// read off the defining operations -- what can be proved is that the two
// windows are apart, and here they are not.
// CHECK-LABEL: func.func @overlapping_windows
// CHECK:         linalg.generic
// CHECK-NOT:     memref.copy
func.func @overlapping_windows(%arena: memref<2048xi8>) {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %a = memref.view %arena[%c0][] : memref<2048xi8> to memref<16x32xi8>
  %b = memref.view %arena[%c256][] : memref<2048xi8> to memref<16x32xi8>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%a : memref<16x32xi8>) outs(%b : memref<16x32xi8>) {
  ^bb0(%v: i8, %o: i8):
    linalg.yield %v : i8
  }
  return
}

// Two windows of the same arena that are apart are fine.
// CHECK-LABEL: func.func @disjoint_windows
// CHECK:         memref.copy
func.func @disjoint_windows(%arena: memref<2048xi8>) {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %a = memref.view %arena[%c0][] : memref<2048xi8> to memref<16x32xi8>
  %b = memref.view %arena[%c512][] : memref<2048xi8> to memref<16x32xi8>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%a : memref<16x32xi8>) outs(%b : memref<16x32xi8>) {
  ^bb0(%v: i8, %o: i8):
    linalg.yield %v : i8
  }
  return
}

// A run of a single byte is not worth a `memcpy` call each: this one is packed
// in the destination and strided in the source, so the longest run common to
// both is one element.
// CHECK-LABEL: func.func @run_too_short
// CHECK:         linalg.generic
// CHECK-NOT:     memref.copy
#stride2 = affine_map<(d0, d1) -> (d0, d1 * 2)>
func.func @run_too_short(%in: memref<16x64xi8>, %out: memref<16x32xi8>) {
  linalg.generic {indexing_maps = [#stride2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%in : memref<16x64xi8>) outs(%out : memref<16x32xi8>) {
  ^bb0(%v: i8, %o: i8):
    linalg.yield %v : i8
  }
  return
}

// Sixteen i32 is 64 bytes of run, which is worth it.
// CHECK-LABEL: func.func @wide_elements
// CHECK:         memref.copy
#rowstride = affine_map<(d0, d1) -> (d0 * 2, d1)>
func.func @wide_elements(%in: memref<32x16xi32>, %out: memref<16x16xi32>) {
  linalg.generic {indexing_maps = [#rowstride, #id2], iterator_types = ["parallel","parallel"]}
    ins(%in : memref<32x16xi32>) outs(%out : memref<16x16xi32>) {
  ^bb0(%v: i32, %o: i32):
    linalg.yield %v : i32
  }
  return
}

// A window narrower than its buffer, with nothing above the packed axis: the
// run is the whole row and there is exactly one axis left to walk. The runtime
// takes that axis in registers and never reaches its carry loop, so this is
// the shape that says so.
// CHECK-LABEL: func.func @one_axis_left
// CHECK:         memref.copy %arg0, %arg1
// CHECK-NOT:     linalg.generic
func.func @one_axis_left(%in: memref<4x8xi8, strided<[16, 1]>>, %out: memref<4x8xi8>) {
  linalg.generic {indexing_maps = [#id2, #id2],
                  iterator_types = ["parallel", "parallel"]}
    ins(%in : memref<4x8xi8, strided<[16, 1]>>) outs(%out : memref<4x8xi8>) {
  ^bb0(%v: i8, %o: i8):
    linalg.yield %v : i8
  }
  return
}

// The same copy arrives written either way round. A channel shuffle comes out
// with the read straight and the permutation on the *write*, which is the same
// copy with the loops named differently -- and whose innermost image is 1024
// contiguous bytes on both sides. Without relabelling by the write's inverse it
// stayed a scalar loop over 4096 elements.
#id5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#swap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>

// CHECK-LABEL: func.func @shuffle_written_backwards
// CHECK:         %[[V:.*]] = memref.reinterpret_cast %arg0
// CHECK-SAME:      sizes: [1, 8, 2, 16, 16]
// CHECK-SAME:      strides: [4096, 256, 2048, 16, 1]
// CHECK:         memref.copy %[[V]], %arg1
// CHECK-NOT:     linalg.generic
func.func @shuffle_written_backwards(%in: memref<1x2x8x16x16xf32>,
                                     %out: memref<1x8x2x16x16xf32>) {
  linalg.generic {indexing_maps = [#id5, #swap],
                  iterator_types = ["parallel","parallel","parallel","parallel","parallel"]}
    ins(%in : memref<1x2x8x16x16xf32>) outs(%out : memref<1x8x2x16x16xf32>) {
  ^bb0(%v: f32, %o: f32):
    linalg.yield %v : f32
  }
  return
}

// A write map that is not a permutation -- a broadcast, say -- is not a copy
// between two views of the same shape.
// CHECK-LABEL: func.func @write_is_a_broadcast
// CHECK:         linalg.generic
// CHECK-NOT:     memref.copy
#drop = affine_map<(d0, d1) -> (d0)>
func.func @write_is_a_broadcast(%in: memref<8x4xf32>, %out: memref<8xf32>) {
  linalg.generic {indexing_maps = [#id2, #drop], iterator_types = ["parallel","parallel"]}
    ins(%in : memref<8x4xf32>) outs(%out : memref<8xf32>) {
  ^bb0(%v: f32, %o: f32):
    linalg.yield %v : f32
  }
  return
}
