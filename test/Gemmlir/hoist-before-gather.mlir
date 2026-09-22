// `elementwise(gather(x))` and `gather(elementwise(x))` compute the same values,
// but the second does the work on the unexpanded operand. After
// --conv-to-img2col that is the difference between quantizing nine elements and
// quantizing one, and between packing f32 and packing i8.

// RUN: gemmlir-opt --hoist-elementwise-before-gather %s | FileCheck %s

#gather = affine_map<(d0,d1) -> (d1 floordiv 4, d1 mod 4)>
#id2    = affine_map<(d0,d1) -> (d0, d1)>

// The conversion moves onto the 4x4 source, and the gather then moves i8.
// CHECK-LABEL: func.func @through_gather
// CHECK:         tensor.empty() : tensor<4x4xi8>
// CHECK:         linalg.generic {{.*}} ins(%arg0 : tensor<4x4xf32>)
// CHECK:         tensor.empty() : tensor<2x16xi8>
// CHECK:         linalg.generic {{.*}} ins(%{{.*}} : tensor<4x4xi8>)
func.func @through_gather(%x: tensor<4x4xf32>) -> tensor<2x16xi8> {
  %e = tensor.empty() : tensor<2x16xf32>
  %g = linalg.generic {indexing_maps = [#gather, #id2], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<4x4xf32>) outs(%e : tensor<2x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<2x16xf32>
  %e2 = tensor.empty() : tensor<2x16xi8>
  %q = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%g : tensor<2x16xf32>) outs(%e2 : tensor<2x16xi8>) {
  ^bb0(%in: f32, %o: i8):
    %i = arith.fptosi %in : f32 to i8
    linalg.yield %i : i8
  } -> tensor<2x16xi8>
  return %q : tensor<2x16xi8>
}

// im2col's result is reshaped before it is used, so the pass looks through a
// single collapse and rebuilds it on the converted values.
// CHECK-LABEL: func.func @through_reshape
// CHECK:         linalg.generic {{.*}} ins(%arg0 : tensor<4x4xf32>)
// CHECK:         linalg.generic {{.*}} ins(%{{.*}} : tensor<4x4xi8>)
// CHECK:         tensor.collapse_shape {{.*}} into tensor<32xi8>
func.func @through_reshape(%x: tensor<4x4xf32>) -> tensor<32xi8> {
  %e = tensor.empty() : tensor<2x16xf32>
  %g = linalg.generic {indexing_maps = [#gather, #id2], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<4x4xf32>) outs(%e : tensor<2x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<2x16xf32>
  %c = tensor.collapse_shape %g [[0, 1]] : tensor<2x16xf32> into tensor<32xf32>
  %e2 = tensor.empty() : tensor<32xi8>
  %q = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
                       iterator_types = ["parallel"]}
    ins(%c : tensor<32xf32>) outs(%e2 : tensor<32xi8>) {
  ^bb0(%in: f32, %o: i8):
    %i = arith.fptosi %in : f32 to i8
    linalg.yield %i : i8
  } -> tensor<32xi8>
  return %q : tensor<32xi8>
}

// Another user of the gathered values means moving the work would duplicate it.
// CHECK-LABEL: func.func @shared
// CHECK:         linalg.generic {{.*}} ins(%arg0 : tensor<4x4xf32>) outs(%{{.*}} : tensor<2x16xf32>)
func.func @shared(%x: tensor<4x4xf32>) -> (tensor<2x16xi8>, tensor<2x16xf32>) {
  %e = tensor.empty() : tensor<2x16xf32>
  %g = linalg.generic {indexing_maps = [#gather, #id2], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<4x4xf32>) outs(%e : tensor<2x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<2x16xf32>
  %e2 = tensor.empty() : tensor<2x16xi8>
  %q = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%g : tensor<2x16xf32>) outs(%e2 : tensor<2x16xi8>) {
  ^bb0(%in: f32, %o: i8):
    %i = arith.fptosi %in : f32 to i8
    linalg.yield %i : i8
  } -> tensor<2x16xi8>
  return %q, %g : tensor<2x16xi8>, tensor<2x16xf32>
}

// A pad only moves data too, and a convolution's padding bufferizes into a fill
// of the whole padded buffer plus a copy of the real input into the middle of
// it. Quantizing first means both of those run on i8 -- a quarter of the memory
// traffic -- and the conversion covers the real input rather than the padded
// one: 972 elements to 768 on the CNN.
//
// The read map is a permutation here, because the conversion has already
// absorbed the layout rewrite's transpose; the padding moves with it, so
// [0, 0, 1, 1] on NCHW becomes [0, 1, 1, 0] on NHWC.
// CHECK-LABEL: func.func @quantize_before_pad
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x3x16x16xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x16x16x3xi8>)
// CHECK:         tensor.pad %[[Q]] low[0, 1, 1, 0] high[0, 1, 1, 0]
// CHECK:           tensor.yield %c0_i8
#perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @quantize_before_pad(%x: tensor<1x3x16x16xf32>) -> tensor<1x18x18x3xi8> {
  %z = arith.constant 0.0 : f32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %p = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %z : f32
  } : tensor<1x3x16x16xf32> to tensor<1x3x18x18xf32>
  %o = tensor.empty() : tensor<1x18x18x3xi8>
  %q = linalg.generic {indexing_maps = [#perm, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%p : tensor<1x3x18x18xf32>) outs(%o : tensor<1x18x18x3xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %i, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x18x18x3xi8>
  return %q : tensor<1x18x18x3xi8>
}

// Adding a constant does not leave zero where it is -- so the new padding value
// is not zero either, it is what the body makes of the old one. The operation
// still moves; only the scalar it writes into the border changes.
// CHECK-LABEL: func.func @shifting_body_moves
// CHECK:         %[[V:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x4xf32>)
// CHECK:         tensor.pad %[[Q]] low[0, 1] high[0, 1]
// CHECK:           tensor.yield %[[V]] : f32
func.func @shifting_body_moves(%x: tensor<1x4xf32>) -> tensor<1x6xf32> {
  %z = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %p = tensor.pad %x low[0, 1] high[0, 1] {
  ^bb0(%a: index, %b: index):
    tensor.yield %z : f32
  } : tensor<1x4xf32> to tensor<1x6xf32>
  %o = tensor.empty() : tensor<1x6xf32>
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel","parallel"]}
    ins(%p : tensor<1x6xf32>) outs(%o : tensor<1x6xf32>) {
  ^bb0(%v: f32, %out: f32):
    %a = arith.addf %v, %one : f32
    linalg.yield %a : f32
  } -> tensor<1x6xf32>
  return %q : tensor<1x6xf32>
}

// `torch.cat` is what an Inception block, a DenseNet layer and a detection neck
// are joined with, and a frontend puts it *before* the requantization: the
// branches' tails stay f32, so neither convolution ends in a requantization and
// neither folds. A concatenation only moves elements, so an elementwise
// operation distributes over it -- the pieces are quantized instead of the
// join, which is also a quarter of the bytes to move.
// CHECK-LABEL: func.func @hoist_over_concat
// CHECK:         %[[A:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x4x4x8xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x4x4x8xi8>)
// CHECK:         %[[B:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg1 : tensor<1x4x4x8xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x4x4x8xi8>)
// CHECK:         tensor.concat dim(3) %[[A]], %[[B]]
// CHECK-NOT:     linalg.generic
func.func @hoist_over_concat(%a: tensor<1x4x4x8xf32>, %b: tensor<1x4x4x8xf32>)
    -> tensor<1x4x4x16xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %j = tensor.concat dim(3) %a, %b
       : (tensor<1x4x4x8xf32>, tensor<1x4x4x8xf32>) -> tensor<1x4x4x16xf32>
  %e = tensor.empty() : tensor<1x4x4x16xi8>
  %q = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                                        affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%j : tensor<1x4x4x16xf32>) outs(%e : tensor<1x4x4x16xi8>) {
  ^bb0(%x: f32, %o: i8):
    %d = arith.divf %x, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x4x4x16xi8>
  return %q : tensor<1x4x4x16xi8>
}

// A slice of an elementwise result is that operation over the slice. Where the
// channels are split -- ShuffleNet's unit passes half of them through --
// --share-branch-quantization leaves the other half reading the shared
// activation back through a dequantization and a relayout over the *whole*
// tensor before slicing. Taking the slice first does the work on the half that
// is wanted: 2048 elements instead of 4096 + 4096 + 2048.
// CHECK-LABEL: func.func @slice_before_elementwise
// CHECK:         %[[S:.*]] = tensor.extract_slice %arg0[0, 0, 0, 8] [1, 4, 4, 8] [1, 1, 1, 1]
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[S]] : tensor<1x4x4x8xi8>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x4x4x8xf32>)
// CHECK-NOT:     tensor<1x4x4x16xf32>
func.func @slice_before_elementwise(%q: tensor<1x4x4x16xi8>) -> tensor<1x4x4x8xf32> {
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<1x4x4x16xf32>
  %d = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                                        affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%q : tensor<1x4x4x16xi8>) outs(%e : tensor<1x4x4x16xf32>) {
  ^bb0(%x: i8, %o: f32):
    %w = arith.sitofp %x : i8 to f32
    %m = arith.mulf %w, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x4x4x16xf32>
  %sl = tensor.extract_slice %d[0, 0, 0, 8] [1, 4, 4, 8] [1, 1, 1, 1]
        : tensor<1x4x4x16xf32> to tensor<1x4x4x8xf32>
  return %sl : tensor<1x4x4x8xf32>
}

// A transpose only renames the axes, so the slice moves with them.
// CHECK-LABEL: func.func @slice_before_transpose
// CHECK:         %[[S:.*]] = tensor.extract_slice %arg0[0, 0, 0, 8] [1, 4, 4, 8] [1, 1, 1, 1]
// CHECK:         linalg.transpose ins(%[[S]] : tensor<1x4x4x8xf32>)
func.func @slice_before_transpose(%x: tensor<1x4x4x16xf32>) -> tensor<1x8x4x4xf32> {
  %e = tensor.empty() : tensor<1x16x4x4xf32>
  %t = linalg.transpose ins(%x : tensor<1x4x4x16xf32>) outs(%e : tensor<1x16x4x4xf32>)
       permutation = [0, 3, 1, 2]
  %sl = tensor.extract_slice %t[0, 8, 0, 0] [1, 8, 4, 4] [1, 1, 1, 1]
        : tensor<1x16x4x4xf32> to tensor<1x8x4x4xf32>
  return %sl : tensor<1x8x4x4xf32>
}

// A transposing requantization on a matmul's accumulator is taken apart: the
// conversion moves into the layout the accelerator writes, where it folds, and
// the permutation is left as a copy of i8 rather than of i32.
//
// Only there. Anywhere else the permutation is a layout rewrite with nothing to
// fold into, and splitting it puts a copy between a convolution and its input.
// CHECK-LABEL: func.func @split_transpose_out
// CHECK:         %[[M:.*]] = linalg.matmul
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$ID:.*]], #[[$ID]]]
// CHECK-SAME:      ins(%[[M]] : tensor<16x32xi32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<16x32xi8>)
// CHECK:           arith.trunci
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[R]] : tensor<16x32xi8>)
// CHECK-SAME:      outs(%{{.*}} : tensor<32x16xi8>)
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: i8, %{{.*}}: i8):
// CHECK-NEXT:      linalg.yield %[[IN]]
#tp = affine_map<(d0, d1) -> (d1, d0)>
func.func @split_transpose_out(%a: tensor<16x64xi8>, %b: tensor<64x32xi8>)
    -> tensor<32x16xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %z = arith.constant 0 : i32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %i = tensor.empty() : tensor<16x32xi32>
  %f = linalg.fill ins(%z : i32) outs(%i : tensor<16x32xi32>) -> tensor<16x32xi32>
  %acc = linalg.matmul ins(%a, %b : tensor<16x64xi8>, tensor<64x32xi8>)
    outs(%f : tensor<16x32xi32>) -> tensor<16x32xi32>
  %e = tensor.empty() : tensor<32x16xi8>
  %r = linalg.generic {indexing_maps = [#tp, #id2], iterator_types = ["parallel","parallel"]}
    ins(%acc : tensor<16x32xi32>) outs(%e : tensor<32x16xi8>) {
  ^bb0(%in: i32, %o: i8):
    %ff = arith.sitofp %in : i32 to f32
    %m = arith.mulf %ff, %s : f32
    %rd = math.roundeven %m : f32
    %ii = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %ii, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<32x16xi8>
  return %r : tensor<32x16xi8>
}

// Not on anything else: this one's producer is the block argument.
// CHECK-LABEL: func.func @no_split_without_contraction
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<16x32xi32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<32x16xi8>)
// CHECK-NOT:     linalg.generic
func.func @no_split_without_contraction(%a: tensor<16x32xi32>) -> tensor<32x16xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<32x16xi8>
  %r = linalg.generic {indexing_maps = [#tp, #id2], iterator_types = ["parallel","parallel"]}
    ins(%a : tensor<16x32xi32>) outs(%e : tensor<32x16xi8>) {
  ^bb0(%in: i32, %o: i8):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %s : f32
    %rd = math.roundeven %m : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<32x16xi8>
  return %r : tensor<32x16xi8>
}

// A frontend writes a constant 0 rather than the dimension wherever an axis has
// extent 1, so an NCHW-to-NHWC relayout on a batch of one is not a permutation
// as far as `AffineMap::isPermutation` is concerned. Refusing those maps left
// the requantization on the far side of every grouped convolution's join, and
// with it every one of the G convolutions unfolded.
// CHECK-LABEL: func.func @concat_with_a_unit_batch
// CHECK:         %[[A:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x4x4x2xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x4x4xi8>)
// CHECK:         %[[B:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg1 : tensor<1x4x4x2xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x4x4xi8>)
// CHECK:         tensor.concat dim(1) %[[A]], %[[B]]
#unit = affine_map<(d0, d1, d2, d3) -> (0, d2, d3, d1)>
func.func @concat_with_a_unit_batch(%x: tensor<1x4x4x2xf32>, %y: tensor<1x4x4x2xf32>)
    -> tensor<1x4x4x4xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %j = tensor.concat dim(3) %x, %y : (tensor<1x4x4x2xf32>, tensor<1x4x4x2xf32>) -> tensor<1x4x4x4xf32>
  %e = tensor.empty() : tensor<1x4x4x4xi8>
  %q = linalg.generic {indexing_maps = [#unit, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%j : tensor<1x4x4x4xf32>) outs(%e : tensor<1x4x4x4xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %s : f32
    %i = arith.fptosi %d : f32 to i8
    linalg.yield %i : i8
  } -> tensor<1x4x4x4xi8>
  return %q : tensor<1x4x4x4xi8>
}

// A grouped convolution's shared dequantization has one slice per group hanging
// off it, and every one of those slices is followed by a quantization at the
// *same* scale -- `--share-branch-quantization` gives them one. Left alone that
// is an identity round trip through f32 over the whole activation: 10368
// elements out and 4 x 2592 back on `gmid`, `gmin` and `gdown`.
//
// One use is not the only case worth moving: a set of slices that between them
// ask for no more than the whole replaces one pass over everything with one
// pass over each part, and the producer then dies.
// Each slice ends up on the i8 input with its own half-sized conversion; the
// round trip itself collapses later, when fusion puts each quantization back
// against its dequantization and `RequantizeByOneIsACopy` sees the scales are
// equal.
// CHECK-LABEL: func.func @slices_of_a_shared_producer
// CHECK:         tensor.extract_slice %arg0[0, 0] [16, 16]
// CHECK-SAME:      tensor<16x32xi8> to tensor<16x16xi8>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}} : tensor<16x16xi8>)
// CHECK:         tensor.extract_slice %arg0[0, 16] [16, 16]
// CHECK-SAME:      tensor<16x32xi8> to tensor<16x16xi8>
#idq = affine_map<(d0, d1) -> (d0, d1)>
func.func @slices_of_a_shared_producer(%x: tensor<16x32xi8>)
    -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<16x32xf32>
  %d = linalg.generic {indexing_maps = [#idq, #idq], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<16x32xi8>) outs(%e : tensor<16x32xf32>) {
  ^bb0(%v: i8, %o: f32):
    %i = arith.extsi %v : i8 to i32
    %f = arith.sitofp %i : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<16x32xf32>
  %a = tensor.extract_slice %d[0, 0] [16, 16] [1, 1] : tensor<16x32xf32> to tensor<16x16xf32>
  %b = tensor.extract_slice %d[0, 16] [16, 16] [1, 1] : tensor<16x32xf32> to tensor<16x16xf32>
  return %a, %b : tensor<16x16xf32>, tensor<16x16xf32>
}

// Not when what feeds it is an accelerator layer: splitting the pass
// restructures everything between here and there, and where that reaches a
// convolution the convolution stops folding. `grp` traded a `conv2d_i8` for a
// `matmul_i8` and went 19.3 -> 24.8 ms.
// CHECK-LABEL: func.func @not_below_a_convolution
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}} : tensor<1x16x16x32xi32>)
// CHECK:         tensor.extract_slice
#id4b = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @not_below_a_convolution(%in: tensor<1x16x16x16xi8>, %w: tensor<1x1x16x32xi8>,
                                   %init: tensor<1x16x16x32xi32>)
    -> (tensor<1x16x16x8xf32>, tensor<1x16x16x8xf32>) {
  %s = arith.constant 2.000000e-02 : f32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %w : tensor<1x16x16x16xi8>, tensor<1x1x16x32xi8>) outs(%init : tensor<1x16x16x32xi32>) -> tensor<1x16x16x32xi32>
  %e = tensor.empty() : tensor<1x16x16x32xf32>
  %d = linalg.generic {indexing_maps = [#id4b, #id4b], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x16x16x32xi32>) outs(%e : tensor<1x16x16x32xf32>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x16x16x32xf32>
  %a = tensor.extract_slice %d[0, 0, 0, 0] [1, 16, 16, 8] [1, 1, 1, 1] : tensor<1x16x16x32xf32> to tensor<1x16x16x8xf32>
  %b = tensor.extract_slice %d[0, 0, 0, 8] [1, 16, 16, 8] [1, 1, 1, 1] : tensor<1x16x16x32xf32> to tensor<1x16x16x8xf32>
  return %a, %b : tensor<1x16x16x8xf32>, tensor<1x16x16x8xf32>
}

// Overlapping slices would ask for more than the whole, so each one would be
// recomputed.
// CHECK-LABEL: func.func @overlapping_slices
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<16x32xi8>)
func.func @overlapping_slices(%x: tensor<16x32xi8>)
    -> (tensor<16x24xf32>, tensor<16x24xf32>) {
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<16x32xf32>
  %d = linalg.generic {indexing_maps = [#idq, #idq], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<16x32xi8>) outs(%e : tensor<16x32xf32>) {
  ^bb0(%v: i8, %o: f32):
    %i = arith.extsi %v : i8 to i32
    %f = arith.sitofp %i : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<16x32xf32>
  %a = tensor.extract_slice %d[0, 0] [16, 24] [1, 1] : tensor<16x32xf32> to tensor<16x24xf32>
  %b = tensor.extract_slice %d[0, 8] [16, 24] [1, 1] : tensor<16x32xf32> to tensor<16x24xf32>
  return %a, %b : tensor<16x24xf32>, tensor<16x24xf32>
}

// Across a padding, only in the narrowing direction. The point of moving an
// operation before a `tensor.pad` is that the fill and the copy it bufferizes
// into should run on the *smaller* type; moving a dequantization the same way
// makes the padded buffer four times the bytes instead.
// CHECK-LABEL: func.func @no_widening_across_a_pad
// CHECK:         tensor.pad
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%{{.*}} : tensor<1x18x18x8xf32>)
func.func @no_widening_across_a_pad(%x: tensor<1x16x16x8xi8>) -> tensor<1x18x18x8xf32> {
  %z = arith.constant 0 : i8
  %s = arith.constant 2.000000e-02 : f32
  %p = tensor.pad %x low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %z : i8
  } : tensor<1x16x16x8xi8> to tensor<1x18x18x8xi8>
  %e = tensor.empty() : tensor<1x18x18x8xf32>
  %d = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%p : tensor<1x18x18x8xi8>) outs(%e : tensor<1x18x18x8xf32>) {
  ^bb0(%v: i8, %o: f32):
    %i = arith.extsi %v : i8 to i32
    %f = arith.sitofp %i : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x18x18x8xf32>
  return %d : tensor<1x18x18x8xf32>
}

// A grouped convolution pads its input on the two spatial axes and then takes
// one channel slice per group. Those commute exactly, and taken in the other
// order whatever quantization sits on the slice can move in front of the
// padding -- so the fill and the copy a padding bufferizes into run on i8
// instead of f32. On `gmid` the padding was 10368 f32 elements filled and an
// 8192-element f32 copy into the middle of them.
// CHECK-LABEL: func.func @slice_before_pad
// CHECK:         %[[S:.*]] = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 16, 16, 8]
// CHECK:         tensor.pad %[[S]] low[0, 1, 1, 0] high[0, 1, 1, 0]
// CHECK-NOT:     tensor.extract_slice
func.func @slice_before_pad(%x: tensor<1x16x16x32xf32>) -> tensor<1x18x18x8xf32> {
  %z = arith.constant 0.0 : f32
  %p = tensor.pad %x low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %z : f32
  } : tensor<1x16x16x32xf32> to tensor<1x18x18x32xf32>
  %s = tensor.extract_slice %p[0, 0, 0, 0] [1, 18, 18, 8] [1, 1, 1, 1]
    : tensor<1x18x18x32xf32> to tensor<1x18x18x8xf32>
  return %s : tensor<1x18x18x8xf32>
}

// A slice that cuts into a padded axis does not commute with the padding: it
// would take the border with it.
// CHECK-LABEL: func.func @slice_cuts_the_border
// CHECK:         tensor.pad %arg0
// CHECK:         tensor.extract_slice
func.func @slice_cuts_the_border(%x: tensor<1x16x16x32xf32>) -> tensor<1x8x18x32xf32> {
  %z = arith.constant 0.0 : f32
  %p = tensor.pad %x low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %z : f32
  } : tensor<1x16x16x32xf32> to tensor<1x18x18x32xf32>
  %s = tensor.extract_slice %p[0, 0, 0, 0] [1, 8, 18, 32] [1, 1, 1, 1]
    : tensor<1x18x18x32xf32> to tensor<1x8x18x32xf32>
  return %s : tensor<1x8x18x32xf32>
}

// Slices that between them ask for more than the whole would pad each of them
// over again.
// CHECK-LABEL: func.func @overlapping_slices_of_a_pad
// CHECK:         tensor.pad %arg0
// CHECK:         tensor.extract_slice
// CHECK:         tensor.extract_slice
func.func @overlapping_slices_of_a_pad(%x: tensor<1x16x16x32xf32>)
    -> (tensor<1x18x18x24xf32>, tensor<1x18x18x24xf32>) {
  %z = arith.constant 0.0 : f32
  %p = tensor.pad %x low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %z : f32
  } : tensor<1x16x16x32xf32> to tensor<1x18x18x32xf32>
  %a = tensor.extract_slice %p[0, 0, 0, 0] [1, 18, 18, 24] [1, 1, 1, 1]
    : tensor<1x18x18x32xf32> to tensor<1x18x18x24xf32>
  %b = tensor.extract_slice %p[0, 0, 0, 8] [1, 18, 18, 24] [1, 1, 1, 1]
    : tensor<1x18x18x32xf32> to tensor<1x18x18x24xf32>
  return %a, %b : tensor<1x18x18x24xf32>, tensor<1x18x18x24xf32>
}

// -----

// The padding value goes through the operation like everything else:
// `q(pad(x, v))` is `pad(q(x), q(v))`, exactly, because a padding only writes
// `v` or copies `x`. A **max-pool pads with -inf**, and that is the case this
// covers -- the quantization clamps it to -128, and the padded buffer is i8
// rather than four times the bytes in f32.

#permn = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#idn   = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @minus_inf_pad
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x8x6x6xf32>)
// CHECK:         } -> tensor<1x6x6x8xi8>
// The scalar travels the same body: -inf over the scale, rounded, clamped.
// CHECK:         %[[V:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK:         tensor.pad %[[Q]] low[0, 1, 1, 0] high[0, 1, 1, 0]
// CHECK:           tensor.yield %[[V]] : i8
// CHECK:         tensor<1x6x6x8xi8> to tensor<1x8x8x8xi8>
func.func @minus_inf_pad(%x: tensor<1x8x6x6xf32>) -> tensor<1x8x8x8xi8> {
  %ninf = arith.constant 0xFF800000 : f32
  %scale = arith.constant 0.05 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %p = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %ninf : f32
  } : tensor<1x8x6x6xf32> to tensor<1x8x8x8xf32>
  %e = tensor.empty() : tensor<1x8x8x8xi8>
  %q = linalg.generic {indexing_maps = [#permn, #idn],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%p : tensor<1x8x8x8xf32>) outs(%e : tensor<1x8x8x8xi8>) {
  ^bb0(%in: f32, %out: i8):
    %0 = arith.divf %in, %scale : f32
    %1 = math.roundeven %0 : f32
    %2 = arith.fptosi %1 : f32 to i32
    %3 = arith.maxsi %2, %lo : i32
    %4 = arith.minsi %3, %hi : i32
    %5 = arith.trunci %4 : i32 to i8
    linalg.yield %5 : i8
  } -> tensor<1x8x8x8xi8>
  return %q : tensor<1x8x8x8xi8>
}

// -----

// A relayout does not care whether an element is padding or image, so the pad
// moves below it with its own axes permuted the same way. That is what puts a
// padded max-pool's padding next to the quantization -- the layout rewrite
// turns the pool into NHWC and leaves the frontend's pad in NCHW.

// CHECK-LABEL: func.func @pad_below_transpose
// CHECK:         %[[T:.*]] = linalg.transpose ins(%arg0 : tensor<1x8x6x6xf32>)
// CHECK-SAME:      permutation = [0, 2, 3, 1]
// CHECK:         tensor.pad %[[T]] low[0, 1, 1, 0] high[0, 1, 1, 0]
// CHECK:         tensor<1x6x6x8xf32> to tensor<1x8x8x8xf32>
func.func @pad_below_transpose(%x: tensor<1x8x6x6xf32>) -> tensor<1x8x8x8xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %p = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %ninf : f32
  } : tensor<1x8x6x6xf32> to tensor<1x8x8x8xf32>
  %e = tensor.empty() : tensor<1x8x8x8xf32>
  %t = linalg.transpose ins(%p : tensor<1x8x8x8xf32>) outs(%e : tensor<1x8x8x8xf32>)
       permutation = [0, 2, 3, 1]
  return %t : tensor<1x8x8x8xf32>
}

// -----

// A pad whose value is not a single constant -- the region yields something
// that depends on where it is -- has no scalar to move.

// CHECK-LABEL: func.func @position_dependent_pad
// CHECK:         tensor.pad %arg0
// CHECK:         linalg.transpose
func.func @position_dependent_pad(%x: tensor<1x8x6x6xf32>) -> tensor<1x8x8x8xf32> {
  %e = tensor.empty() : tensor<1x8x8x8xf32>
  %p = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    %i = arith.index_cast %c : index to i32
    %f = arith.sitofp %i : i32 to f32
    tensor.yield %f : f32
  } : tensor<1x8x6x6xf32> to tensor<1x8x8x8xf32>
  %t = linalg.transpose ins(%p : tensor<1x8x8x8xf32>) outs(%e : tensor<1x8x8x8xf32>)
       permutation = [0, 2, 3, 1]
  return %t : tensor<1x8x8x8xf32>
}

// -----

// Gemmini's pooling pads with **zero**; PyTorch pads a max-pool with -inf. Under
// a relu the two agree -- the padding loses to any real element, and every one
// of them is at least zero -- so the padding is normalised to zero here, where
// the relu is still visible. That is what lets the pool fold into the
// convolution's `pool_padding` later.

// CHECK-LABEL: func.func @max_pool_pad_under_relu
// CHECK:         %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         tensor.pad
// CHECK:           tensor.yield %[[Z]] : f32
// CHECK:         linalg.pooling_nhwc_max
func.func @max_pool_pad_under_relu(%acc: tensor<1x8x8x4xf32>) -> tensor<1x4x4x4xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x8x8x4xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                          iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc : tensor<1x8x8x4xf32>) outs(%e : tensor<1x8x8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %c, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x4xf32>
  %p = tensor.pad %relu low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %ninf : f32
  } : tensor<1x8x8x4xf32> to tensor<1x10x10x4xf32>
  %w = tensor.empty() : tensor<3x3xf32>
  %o = tensor.empty() : tensor<1x4x4x4xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%o : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %pool = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%p, %w : tensor<1x10x10x4xf32>, tensor<3x3xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %pool : tensor<1x4x4x4xf32>
}

// -----

// Without the relu there is nothing to say the padding loses, so -inf stays.

// CHECK-LABEL: func.func @max_pool_pad_without_relu
// CHECK:         tensor.pad
// CHECK:           tensor.yield %[[N:.*]] : f32
// CHECK-NOT:     arith.constant 0.000000e+00
func.func @max_pool_pad_without_relu(%x: tensor<1x8x8x4xf32>) -> tensor<1x4x4x4xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %p = tensor.pad %x low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %ninf : f32
  } : tensor<1x8x8x4xf32> to tensor<1x10x10x4xf32>
  %w = tensor.empty() : tensor<3x3xf32>
  %o = tensor.empty() : tensor<1x4x4x4xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%o : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %pool = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%p, %w : tensor<1x10x10x4xf32>, tensor<3x3xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %pool : tensor<1x4x4x4xf32>
}

// -----

// An Inception block's join has one quantization **per branch** of the next
// block, and nothing merges them: three identical `linalg.generic`s on one
// value. A `hasOneUse` test refuses every one of GoogLeNet's nine joins on
// that. Distributing once and giving each copy the same result is the merge --
// and a global `--cse` is not the way to get it, because it merges the
// contraction tails too, which makes them multi-use and stops
// `matchRequantize` folding any of them.
//
// Two kinds of user are skipped rather than matched, and the join stays for
// them: a `tensor.pad`, which is the pooling branch and ends up quantized below
// the pool anyway, and a `linalg.transpose`, which reads the join in its own
// layout. Requiring every user to match refuses the join outright, and refusing
// is the worse trade.
//
// The pieces are built at the **concatenation**, not at the matched operation:
// the shared result now feeds users that can sit above it.
// CHECK-LABEL: func.func @a_join_with_one_quantization_per_branch
// CHECK:         %[[A:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0
// CHECK:         %[[B:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg1
// CHECK:         %[[C:.*]] = tensor.concat dim(1) %[[A]], %[[B]]
// CHECK-SAME:      -> tensor<1x8xi8>
// Three copies, one result.
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[C]], %[[C]], %[[C]]
func.func @a_join_with_one_quantization_per_branch(%a: tensor<1x4xf32>, %b: tensor<1x4xf32>)
    -> (tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>) {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %j = tensor.concat dim(1) %a, %b : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x8xf32>
  %e0 = tensor.empty() : tensor<1x8xi8>
  %q0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel","parallel"]}
    ins(%j : tensor<1x8xf32>) outs(%e0 : tensor<1x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x8xi8>
  %e1 = tensor.empty() : tensor<1x8xi8>
  %q1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel","parallel"]}
    ins(%j : tensor<1x8xf32>) outs(%e1 : tensor<1x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x8xi8>
  %e2 = tensor.empty() : tensor<1x8xi8>
  %q2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel","parallel"]}
    ins(%j : tensor<1x8xf32>) outs(%e2 : tensor<1x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x8xi8>
  return %q0, %q1, %q2 : tensor<1x8xi8>, tensor<1x8xi8>, tensor<1x8xi8>
}

// -----

// Two quantizations of one join that ask for **different scales** are not one
// operation repeated, and distributing either of them over the pieces would put
// a scale on a branch that did not ask for it. Left alone.
// CHECK-LABEL: func.func @two_scales_on_one_join
// CHECK:         tensor.concat
// CHECK-SAME:      -> tensor<1x8xf32>
func.func @two_scales_on_one_join(%a: tensor<1x4xf32>, %b: tensor<1x4xf32>)
    -> (tensor<1x8xi8>, tensor<1x8xi8>) {
  %s0 = arith.constant 2.000000e-02 : f32
  %s1 = arith.constant 1.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %j = tensor.concat dim(1) %a, %b : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x8xf32>
  %e0 = tensor.empty() : tensor<1x8xi8>
  %q0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel","parallel"]}
    ins(%j : tensor<1x8xf32>) outs(%e0 : tensor<1x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s0 : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x8xi8>
  %e1 = tensor.empty() : tensor<1x8xi8>
  %q1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel","parallel"]}
    ins(%j : tensor<1x8xf32>) outs(%e1 : tensor<1x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s1 : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x8xi8>
  return %q0, %q1 : tensor<1x8xi8>, tensor<1x8xi8>
}

// -----

// An elementwise map absorbs the relayout it reads: walking the transpose's
// *source* through the permutation is the same computation, and an elementwise
// body does not care which order the elements arrive in. One pass over the data
// instead of two.
//
// This is what lets the concatenation rewrite above see an Inception block's
// pooling branch. That branch reads the join, relayouts, pads and pools;
// `--requantize-before-pooling` and `HoistElementwiseBeforePad` walk its
// quantization up to just under the relayout and stop, so the join keeps an f32
// reader and the distribution is refused -- correctly, because letting that
// reader through builds the join **twice**, once in i8 and once in f32, which
// leaves every branch tail with two readers and folding into neither. GoogLeNet
// lost 12 convolutions to scalar loops that way, 61.6 million
// multiply-accumulates.
//
// With this, GoogLeNet's im2col packs go from 24 to 9 and its scalar work from
// 3,350,680 elements to 2,265,240, with the same 59 accelerator calls.
// The map it now reads through is the permutation's inverse: result dimension
// k is the source's perm[k], so iteration dimension k walks source dimension
// perm[k] -- for [0, 2, 3, 1] that is (d0, d1, d2, d3) -> (d0, d3, d1, d2). It
// is printed as an alias, so what the check pins is the *shape*: one generic
// reading the transpose's own source, with no relayout left.
// CHECK-LABEL: func.func @absorb_a_relayout
// CHECK-NOT:     linalg.transpose
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x8x4x4xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x4x4x8xi8>)
func.func @absorb_a_relayout(%x: tensor<1x8x4x4xf32>) -> tensor<1x4x4x8xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x4x4x8xf32>
  %t = linalg.transpose ins(%x : tensor<1x8x4x4xf32>) outs(%e : tensor<1x4x4x8xf32>) permutation = [0, 2, 3, 1]
  %o = tensor.empty() : tensor<1x4x4x8xi8>
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%t : tensor<1x4x4x8xf32>) outs(%o : tensor<1x4x4x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %b = arith.trunci %c1 : i32 to i8
    linalg.yield %b : i8
  } -> tensor<1x4x4x8xi8>
  return %q : tensor<1x4x4x8xi8>
}

// -----

// A relayout with another reader is left alone: absorbing it here would only
// relayout the data twice.
// CHECK-LABEL: func.func @a_shared_relayout_stays
// CHECK:         linalg.transpose
// CHECK:         linalg.generic
func.func @a_shared_relayout_stays(%x: tensor<1x8x4x4xf32>)
    -> (tensor<1x4x4x8xi8>, tensor<1x4x4x8xf32>) {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x4x4x8xf32>
  %t = linalg.transpose ins(%x : tensor<1x8x4x4xf32>) outs(%e : tensor<1x4x4x8xf32>) permutation = [0, 2, 3, 1]
  %o = tensor.empty() : tensor<1x4x4x8xi8>
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%t : tensor<1x4x4x8xf32>) outs(%o : tensor<1x4x4x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %b = arith.trunci %c1 : i32 to i8
    linalg.yield %b : i8
  } -> tensor<1x4x4x8xi8>
  return %q, %t : tensor<1x4x4x8xi8>, tensor<1x4x4x8xf32>
}

// -----

// A map that reads its input **broadcast** does the work once per copy.
// `--fuse-elementwise-around-matmul` has already put the broadcast and the
// elementwise work in one region, which reads a small input through a map that
// drops a dimension and writes the big result. Doing the work first and
// broadcasting the answer is strictly less of it -- and where the input is a
// constant it is none at all, because the folder turns the quantized weight
// into an i8 constant at compile time.
//
// ConvNeXt is the model this is for: its MLP weights reach a
// `linalg.batch_matmul` broadcast into a batch of two, so a 768 x 3072 weight
// is quantized **twice on every inference** -- 81.4 million elements of its
// 83.2 million of scalar work. `--unbatch-single-matmul` reads through such a
// broadcast already, but only for a batch of one, which this is not.
//
// Measured on a probe of the same shape: **18.95 to 9.37 ms**, byte-identical
// to the old object's output and 0 of 40 against the CPU reference.
// CHECK-LABEL: func.func @quantize_before_the_broadcast
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<4x8xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<4x8xi8>)
// CHECK:         arith.trunci
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[Q]] : tensor<4x8xi8>)
// CHECK-SAME:      outs(%{{.*}} : tensor<2x4x8xi8>)
// CHECK-NEXT:    ^bb0
// CHECK-NEXT:      linalg.yield
func.func @quantize_before_the_broadcast(%w: tensor<4x8xf32>) -> tensor<2x4x8xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<2x4x8xi8>
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel","parallel","parallel"]}
    ins(%w : tensor<4x8xf32>) outs(%e : tensor<2x4x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<2x4x8xi8>
  return %q : tensor<2x4x8xi8>
}

// -----

// The broadcast itself has nothing to take out of it, and splitting it would
// produce another one to split: the greedy driver would never return. (That is
// not hypothetical -- the rewrite above writes exactly such a copy.)
// CHECK-LABEL: func.func @a_plain_broadcast_is_left_alone
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic
// CHECK:         return
func.func @a_plain_broadcast_is_left_alone(%w: tensor<4x8xi8>) -> tensor<2x4x8xi8> {
  %e = tensor.empty() : tensor<2x4x8xi8>
  %b = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel","parallel","parallel"]}
    ins(%w : tensor<4x8xi8>) outs(%e : tensor<2x4x8xi8>) {
  ^bb0(%v: i8, %out: i8):
    linalg.yield %v : i8
  } -> tensor<2x4x8xi8>
  return %b : tensor<2x4x8xi8>
}

// -----

// **A join is non-negative when every branch is.** GoogLeNet's pool branch does
// not pool a convolution -- it pools the concatenation of the previous
// inception module's four branches -- so the relu that makes -inf and zero
// agree sits one step further up than the walk used to look, and on four values
// rather than one.
//
// The walk is also *deep*: between the join and the pool there is a relayout on
// each side, and the join's own branches carry their quantization. A budget of
// four steps refused all sixteen of GoogLeNet's pool pads; the note
// [[gemmlir-a-budget-is-not-a-rule]] is about exactly this mistake.
//
// This does **not** yet make the pool fold into a call -- see
// `FoldMaxPoolIntoConv`, which needs one convolution's own output buffer and a
// *symmetric* padding, and a join is neither. It is the precondition, and it is
// sound on its own terms: the padding written is a value the pool's own maximum
// can never prefer.
// CHECK-LABEL: func.func @a_pool_over_a_join
// CHECK:         %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         tensor.pad
// CHECK:           tensor.yield %[[Z]] : f32
func.func @a_pool_over_a_join(%a: tensor<1x8x8x4xf32>, %b: tensor<1x8x8x4xf32>)
    -> tensor<1x8x8x8xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x8x8x4xf32>
  %ra = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%a : tensor<1x8x8x4xf32>) outs(%e : tensor<1x8x8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %c, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x4xf32>
  %rb = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%b : tensor<1x8x8x4xf32>) outs(%e : tensor<1x8x8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %c, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x4xf32>
  %j = tensor.concat dim(3) %ra, %rb
      : (tensor<1x8x8x4xf32>, tensor<1x8x8x4xf32>) -> tensor<1x8x8x8xf32>
  %p = tensor.pad %j low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%i: index, %k: index, %l: index, %m: index):
    tensor.yield %ninf : f32
  } : tensor<1x8x8x8xf32> to tensor<1x10x10x8xf32>
  %w = tensor.empty() : tensor<3x3xf32>
  %o = tensor.empty() : tensor<1x8x8x8xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%o : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  %pool = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%p, %w : tensor<1x10x10x8xf32>, tensor<3x3xf32>) outs(%init : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %pool : tensor<1x8x8x8xf32>
}

// -----

// One branch of the join has no relu, so the join can be negative and the
// padding has to stay what it was.
// CHECK-LABEL: func.func @one_branch_of_the_join_is_signed
// CHECK:         tensor.pad
// CHECK-NOT:     tensor.yield %{{.*}}0.000000e+00
func.func @one_branch_of_the_join_is_signed(%a: tensor<1x8x8x4xf32>, %b: tensor<1x8x8x4xf32>)
    -> tensor<1x8x8x8xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x8x8x4xf32>
  %ra = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%a : tensor<1x8x8x4xf32>) outs(%e : tensor<1x8x8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %c, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x4xf32>
  %j = tensor.concat dim(3) %ra, %b
      : (tensor<1x8x8x4xf32>, tensor<1x8x8x4xf32>) -> tensor<1x8x8x8xf32>
  %p = tensor.pad %j low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%i: index, %k: index, %l: index, %m: index):
    tensor.yield %ninf : f32
  } : tensor<1x8x8x8xf32> to tensor<1x10x10x8xf32>
  %w = tensor.empty() : tensor<3x3xf32>
  %o = tensor.empty() : tensor<1x8x8x8xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%o : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  %pool = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%p, %w : tensor<1x10x10x8xf32>, tensor<3x3xf32>) outs(%init : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %pool : tensor<1x8x8x8xf32>
}

// -----

// A pool of a pool. `max(window)` over non-negative elements is non-negative,
// and so is a padding that is already zero -- which is what this pattern left
// on the pool below. GoogLeNet's stem pools one after the other and needs both
// halves of that.
// CHECK-LABEL: func.func @a_pool_of_a_pool
// CHECK:         %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         tensor.pad
// CHECK:           tensor.yield %[[Z]]
// CHECK:         linalg.pooling_nhwc_max
// CHECK:         tensor.pad
// CHECK:           tensor.yield %[[Z]]
// CHECK:         linalg.pooling_nhwc_max
func.func @a_pool_of_a_pool(%acc: tensor<1x16x16x4xf32>) -> tensor<1x4x4x4xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x16x16x4xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                          iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc : tensor<1x16x16x4xf32>) outs(%e : tensor<1x16x16x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %c, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x16x16x4xf32>
  %p0 = tensor.pad %relu low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%i: index, %k: index, %l: index, %m: index):
    tensor.yield %ninf : f32
  } : tensor<1x16x16x4xf32> to tensor<1x18x18x4xf32>
  %w = tensor.empty() : tensor<3x3xf32>
  %o0 = tensor.empty() : tensor<1x8x8x4xf32>
  %i0 = linalg.fill ins(%ninf : f32) outs(%o0 : tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
  %pool0 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%p0, %w : tensor<1x18x18x4xf32>, tensor<3x3xf32>) outs(%i0 : tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
  %p1 = tensor.pad %pool0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%i: index, %k: index, %l: index, %m: index):
    tensor.yield %ninf : f32
  } : tensor<1x8x8x4xf32> to tensor<1x10x10x4xf32>
  %o1 = tensor.empty() : tensor<1x4x4x4xf32>
  %i1 = linalg.fill ins(%ninf : f32) outs(%o1 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %pool1 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%p1, %w : tensor<1x10x10x4xf32>, tensor<3x3xf32>) outs(%i1 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %pool1 : tensor<1x4x4x4xf32>
}

// -----

// A body that reads its own position in the iteration space cannot be moved
// into a different one. Every pattern in this pass does exactly that -- past a
// pad, a slice, a transpose, a concatenation, a gather -- and a `linalg.index`
// is the one thing whose meaning is the space it sits in.
//
// Not theoretical: an embedding lookup is a `linalg.generic` that reads its
// index, so a decoder-only transformer starts with one. Unguarded, moving it
// tripped MLIR's own folder, which asserts the dimension is in range.

// CHECK-LABEL: func.func @reads_its_index
// CHECK:         linalg.generic
// CHECK:           linalg.index
// CHECK:         tensor.pad
#ridx = affine_map<(d0, d1) -> (d0, d1)>
func.func @reads_its_index(%src: memref<4x8xi32>) -> tensor<6x10xi32> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<4x8xi32>
  %g = linalg.generic {indexing_maps = [#ridx], iterator_types = ["parallel", "parallel"]}
    outs(%e : tensor<4x8xi32>) {
  ^bb0(%o: i32):
    %i = linalg.index 1 : index
    %c = arith.index_cast %i : index to i32
    linalg.yield %c : i32
  } -> tensor<4x8xi32>
  %p = tensor.pad %g low[1, 1] high[1, 1] {
  ^bb0(%a: index, %b: index):
    tensor.yield %z : i32
  } : tensor<4x8xi32> to tensor<6x10xi32>
  return %p : tensor<6x10xi32>
}
