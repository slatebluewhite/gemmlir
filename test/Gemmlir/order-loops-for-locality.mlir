// An elementwise loop nest may run its axes in any order, and the order decides
// how many cache lines it touches. Bufferization leaves them in the
// destination's order, so where the operation also relayouts, the source's
// slowest axis ends up innermost: every load its own cache line, used for four
// of its sixty-four bytes.

// RUN: gemmlir-opt --order-loops-for-locality %s | FileCheck %s
// RUN: gemmlir-opt --order-loops-for-locality --split-input-file %s | FileCheck %s --check-prefix=RELAYOUT

// The quantization at a grouped convolution's input: reads a channel slice of
// an NCHW image, writes the interior of an NHWC padded buffer. The channel axis
// moves the source 1024 bytes and the destination one; the width axis moves
// them four and eight. Width belongs innermost, and then the source is walked
// in its own order.
// CHECK-DAG:   #[[$ID:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:   #[[$P:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: func.func @quantize_and_relayout
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$ID]], #[[$P]]]
#src = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @quantize_and_relayout(%in: memref<1x8x16x16xf32, strided<[8192, 256, 16, 1]>>,
                                 %out: memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>) {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#src, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%in : memref<1x8x16x16xf32, strided<[8192, 256, 16, 1]>>)
    outs(%out : memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>) {
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

// A loop nest that is already in the right order is left exactly as it was --
// the pattern has to fail rather than rewrite, or the greedy driver never
// settles.
// CHECK-LABEL: func.func @already_in_order
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$ID]], #[[$ID]]]
func.func @already_in_order(%in: memref<1x16x16x8xf32>, %out: memref<1x16x16x8xi8>) {
  %s = arith.constant 2.000000e-02 : f32
  linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%in : memref<1x16x16x8xf32>) outs(%out : memref<1x16x16x8xi8>) {
  ^bb0(%v: f32, %o: i8):
    %q = arith.divf %v, %s : f32
    %i = arith.fptosi %q : f32 to i8
    linalg.yield %i : i8
  }
  return
}

// A reduction's axes cannot be permuted freely against its parallel ones, so a
// loop nest that has one is left alone.
// CHECK-LABEL: func.func @has_a_reduction
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "reduction"]
#m2 = affine_map<(d0, d1) -> (d1, d0)>
#m1 = affine_map<(d0, d1) -> (d0)>
func.func @has_a_reduction(%in: memref<64x8xf32>, %out: memref<8xf32>) {
  linalg.generic {indexing_maps = [#m2, #m1], iterator_types = ["parallel","reduction"]}
    ins(%in : memref<64x8xf32>) outs(%out : memref<8xf32>) {
  ^bb0(%v: f32, %o: f32):
    %a = arith.addf %v, %o : f32
    linalg.yield %a : f32
  }
  return
}

// -----

// A relayout written as `linalg.transpose` gets no say in its loop order. The
// named op carries only a permutation, and `--convert-linalg-to-loops` walks
// its iteration space in the **destination's** order -- so the write is
// contiguous and the read jumps a whole channel plane every element:
//
//     flw  fa5,-576(a5)
//     fsw  fa5,-4(a4)
//     addi a4,a4,8        # the write moves 4 bytes an element
//     addi a5,a5,1152     # the read moves 576
//
// GoogLeNet's 1x480x12x12 relayout is exactly that, and program-counter
// sampling makes it the **hottest basic block in the model**. Which side to
// favour is settled and it is the read; writing the transpose out as a generic
// -- the permutation on the read map, identity on the write, a body that yields
// its argument -- is the same operation and puts it in front of that judgement.
//
// The result reads `in[n, c, h, w]` with `w` innermost, so the load is
// contiguous and the store is the scattered one.
// RELAYOUT-DAG:   #[[READ:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// RELAYOUT-DAG:   #[[WRITE:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// RELAYOUT-LABEL: func.func @a_relayout_reads_in_order
// RELAYOUT-NOT:     linalg.transpose
// RELAYOUT:         linalg.generic
// RELAYOUT-SAME:      indexing_maps = [#[[READ]], #[[WRITE]]]
// RELAYOUT-SAME:      ins(%arg0 : memref<1x480x12x12xf32>)
// RELAYOUT-SAME:      outs(%arg1 : memref<1x12x12x480xf32>)
// RELAYOUT-NEXT:    ^bb0(%[[V:.*]]: f32, %{{.*}}: f32):
// RELAYOUT-NEXT:      linalg.yield %[[V]]
func.func @a_relayout_reads_in_order(%in: memref<1x480x12x12xf32>,
                                     %out: memref<1x12x12x480xf32>) {
  linalg.transpose ins(%in : memref<1x480x12x12xf32>)
                   outs(%out : memref<1x12x12x480xf32>) permutation = [0, 2, 3, 1]
  return
}

// -----

// The other direction, NHWC to NCHW, is the same rewrite and the same rule: the
// axis whose read stride is smallest goes innermost. Here that is the channel,
// which is contiguous on the source.
// RELAYOUT-LABEL: func.func @the_other_way_round
// RELAYOUT-NOT:     linalg.transpose
// RELAYOUT:         linalg.generic
// RELAYOUT:           linalg.yield
func.func @the_other_way_round(%in: memref<1x8x8x64xf32>,
                               %out: memref<1x64x8x8xf32>) {
  linalg.transpose ins(%in : memref<1x8x8x64xf32>)
                   outs(%out : memref<1x64x8x8xf32>) permutation = [0, 3, 1, 2]
  return
}

// -----

// On tensors there is no loop order yet to argue about, and rewriting the named
// op there would only take it away from everything that matches a relayout by
// name -- which is most of the front of the pipeline.
// RELAYOUT-LABEL: func.func @a_tensor_transpose_stays
// RELAYOUT:         linalg.transpose
func.func @a_tensor_transpose_stays(%in: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {
  %e = tensor.empty() : tensor<1x8x4x4xf32>
  %t = linalg.transpose ins(%in : tensor<1x4x4x8xf32>)
                        outs(%e : tensor<1x8x4x4xf32>) permutation = [0, 3, 1, 2]
  return %t : tensor<1x8x4x4xf32>
}
