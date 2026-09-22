// Elementwise fusion that leaves the matmuls alone. Fusing a quantization into
// the matmul that consumes it hides the matmul from --convert-linalg-to-gemmlir
// and it stops being offloaded -- with the stock --linalg-fuse-elementwise-ops
// a CNN went from four offloaded matmuls to one, and from 35 ms to 232 ms.

// RUN: gemmlir-opt --fuse-elementwise-around-matmul --canonicalize %s | FileCheck %s
// RUN: gemmlir-opt --linalg-fuse-elementwise-ops --canonicalize %s | FileCheck %s --check-prefix=STOCK

#id = affine_map<(d0, d1) -> (d0, d1)>
#id1 = affine_map<(d0) -> (d0)>

// The two elementwise operations become one, and the matmul still takes its
// operand from a separate operation it can be matched through.
// CHECK-LABEL: func.func @around
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic
// CHECK:         linalg.matmul

// The stock pass pulls the scaling into the matmul's operand instead.
// STOCK-LABEL: func.func @around
// STOCK:         linalg.matmul
// STOCK-NOT:     linalg.generic
func.func @around(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %two = arith.constant 2.0 : f32
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<8x8xf32>

  // scale, then negate: two elementwise passes that should become one
  %s = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%a : tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %m = arith.mulf %x, %two : f32
    linalg.yield %m : f32
  } -> tensor<8x8xf32>
  %n = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%s : tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %g = arith.negf %x : f32
    linalg.yield %g : f32
  } -> tensor<8x8xf32>

  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
  %r = linalg.matmul ins(%n, %b : tensor<8x8xf32>, tensor<8x8xf32>) outs(%f : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// An im2col gather reads a 3x3 neighbourhood, so its iteration space is nine
// times the tensor it reads. Fusing the relu into it would run the relu nine
// times per element -- 2048 elements became 3528 on the CNN -- and would also
// strand the conversion on the far side of the gather, where
// --hoist-elementwise-before-gather can no longer move it. So the relu stays
// where it is and the gather remains a pure copy.
// CHECK-LABEL: func.func @not_into_a_gather
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x8x8xf32>)
// CHECK:           arith.maximumf
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%{{.*}} : tensor<1x18x36xf32>)
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      linalg.yield %[[IN]]

// The stock pass fuses it in, which is what the guard exists to prevent.
// STOCK-LABEL: func.func @not_into_a_gather
// STOCK:         linalg.generic
// STOCK-SAME:      outs(%{{.*}} : tensor<1x18x36xf32>)
// STOCK:           arith.maximumf
#nchw = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// (batch, c*kh*kw, oh*ow) <- (batch, c, oh + kh, ow + kw), a 3x3 im2col gather
#gather = affine_map<(b, k, p) -> (b, k floordiv 9, p floordiv 6 + (k mod 9) floordiv 3, p mod 6 + k mod 3)>
#out = affine_map<(b, k, p) -> (b, k, p)>
func.func @not_into_a_gather(%x: tensor<1x2x8x8xf32>) -> tensor<1x18x36xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x2x8x8xf32>
  %r = linalg.generic {indexing_maps = [#nchw, #nchw], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%x : tensor<1x2x8x8xf32>) outs(%e : tensor<1x2x8x8xf32>) {
  ^bb0(%v: f32, %o: f32):
    %m = arith.maximumf %v, %zero : f32
    linalg.yield %m : f32
  } -> tensor<1x2x8x8xf32>
  %g = tensor.empty() : tensor<1x18x36xf32>
  %c = linalg.generic {indexing_maps = [#gather, #out], iterator_types = ["parallel","parallel","parallel"]}
    ins(%r : tensor<1x2x8x8xf32>) outs(%g : tensor<1x18x36xf32>) {
  ^bb0(%v: f32, %o: f32):
    linalg.yield %v : f32
  } -> tensor<1x18x36xf32>
  return %c : tensor<1x18x36xf32>
}

// A bias arrives broadcast over the whole activation and then reshaped, which
// is how torch-mlir materialises it. Fusion would absorb the broadcast -- the
// iteration spaces match -- but the reshape is in the way, so it is sunk into
// the broadcast first and the two are then adjacent. On the CNN this removed
// the 2048- and 784-element materialisations entirely.
// CHECK-LABEL: func.func @bias_through_reshape
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0, %arg1 : tensor<8x256xf32>, tensor<8xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<8x256xf32>)
// CHECK:           arith.addf
// CHECK-NOT:     linalg.generic
#chan = affine_map<(n, c, h, w) -> (c)>
#nchw4 = affine_map<(n, c, h, w) -> (n, c, h, w)>

func.func @bias_through_reshape(%acc: tensor<8x256xf32>, %bias: tensor<8xf32>) -> tensor<8x256xf32> {
  %e = tensor.empty() : tensor<1x8x16x16xf32>
  %b = linalg.generic {indexing_maps = [#chan, #nchw4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%bias : tensor<8xf32>) outs(%e : tensor<1x8x16x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<1x8x16x16xf32>
  %c = tensor.collapse_shape %b [[0, 1], [2, 3]] : tensor<1x8x16x16xf32> into tensor<8x256xf32>
  %e2 = tensor.empty() : tensor<8x256xf32>
  %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc, %c : tensor<8x256xf32>, tensor<8x256xf32>) outs(%e2 : tensor<8x256xf32>) {
  ^bb0(%a: f32, %bb: f32, %o: f32):
    %s = arith.addf %a, %bb : f32
    linalg.yield %s : f32
  } -> tensor<8x256xf32>
  return %r : tensor<8x256xf32>
}

// The collapsed group merges two dimensions the broadcast actually reads, so
// recovering either index would need floordiv/mod. Left alone.
// CHECK-LABEL: func.func @two_read_dims_in_a_group
// CHECK:         tensor.collapse_shape
#hw = affine_map<(n, c, h, w) -> (h, w)>
func.func @two_read_dims_in_a_group(%m: tensor<16x16xf32>) -> tensor<8x256xf32> {
  %e = tensor.empty() : tensor<1x8x16x16xf32>
  %b = linalg.generic {indexing_maps = [#hw, #nchw4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%m : tensor<16x16xf32>) outs(%e : tensor<1x8x16x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<1x8x16x16xf32>
  %c = tensor.collapse_shape %b [[0, 1], [2, 3]] : tensor<1x8x16x16xf32> into tensor<8x256xf32>
  return %c : tensor<8x256xf32>
}

// The read dimension shares its group with a dimension of extent 4, so the
// collapsed index is not the read index. Left alone.
// CHECK-LABEL: func.func @group_has_another_real_dim
// CHECK:         tensor.collapse_shape
func.func @group_has_another_real_dim(%bias: tensor<8xf32>) -> tensor<32x256xf32> {
  %e = tensor.empty() : tensor<4x8x16x16xf32>
  %b = linalg.generic {indexing_maps = [#chan, #nchw4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%bias : tensor<8xf32>) outs(%e : tensor<4x8x16x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<4x8x16x16xf32>
  %c = tensor.collapse_shape %b [[0, 1], [2, 3]] : tensor<4x8x16x16xf32> into tensor<32x256xf32>
  return %c : tensor<32x256xf32>
}

// The other half of the reshape problem: a frontend reshapes between the 2-D
// form a contraction wants and the 4-D form an activation has, so a dequantize
// lands on 8x256 and the relu that follows it on 1x8x16x16 with a view in
// between. Collapsing the consumer puts them back in the same iteration space
// and they fuse into one loop over one buffer -- 2832 elements on the CNN, and
// an 8 KB f32 temporary. Note the frontend writes a constant 0 rather than the
// dimension where an axis has extent 1, which the match has to allow.
// CHECK-LABEL: func.func @elementwise_over_expand
// CHECK:         %[[G:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<8x256xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<8x256xf32>)
// CHECK:           arith.maximumf
// CHECK:         tensor.expand_shape %[[G]]
// CHECK-NOT:     linalg.generic
#zero0 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
func.func @elementwise_over_expand(%x: tensor<8x256xf32>) -> tensor<1x8x16x16xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.expand_shape %x [[0, 1], [2, 3]] output_shape [1, 8, 16, 16]
       : tensor<8x256xf32> into tensor<1x8x16x16xf32>
  %o = tensor.empty() : tensor<1x8x16x16xf32>
  %r = linalg.generic {indexing_maps = [#zero0, #nchw], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%e : tensor<1x8x16x16xf32>) outs(%o : tensor<1x8x16x16xf32>) {
  ^bb0(%v: f32, %out: f32):
    %m = arith.maximumf %v, %zero : f32
    linalg.yield %m : f32
  } -> tensor<1x8x16x16xf32>
  return %r : tensor<1x8x16x16xf32>
}

// A map that actually permutes is not a view of the same element, so the
// collapsed form would read something else. Left alone.
// CHECK-LABEL: func.func @permuting_map_over_expand
// CHECK:         tensor.expand_shape
// CHECK:         linalg.generic
#swap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
func.func @permuting_map_over_expand(%x: tensor<8x256xf32>) -> tensor<1x8x16x16xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.expand_shape %x [[0, 1], [2, 3]] output_shape [1, 8, 16, 16]
       : tensor<8x256xf32> into tensor<1x8x16x16xf32>
  %o = tensor.empty() : tensor<1x8x16x16xf32>
  %r = linalg.generic {indexing_maps = [#swap, #nchw], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%e : tensor<1x8x16x16xf32>) outs(%o : tensor<1x8x16x16xf32>) {
  ^bb0(%v: f32, %out: f32):
    %m = arith.maximumf %v, %zero : f32
    linalg.yield %m : f32
  } -> tensor<1x8x16x16xf32>
  return %r : tensor<1x8x16x16xf32>
}

// The layout rewrite cancels the transposes it creates against each other, but
// the one in front of the network's input has nothing to cancel against -- and
// the operation that would absorb it, the input's quantization, does not exist
// until the quantization passes have run. So the same pattern runs again here:
// one operation that reads NCHW f32 and writes NHWC i8, instead of a transpose
// and then a conversion. Half the passes over the data.
// CHECK-LABEL: func.func @quantize_absorbs_a_transpose
// CHECK-NOT:     linalg.transpose
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x3x4x4xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x4x4x3xi8>)
// CHECK:           arith.fptosi
#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @quantize_absorbs_a_transpose(%x: tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<1x4x4x3xf32>
  %t = linalg.transpose ins(%x : tensor<1x3x4x4xf32>) outs(%e : tensor<1x4x4x3xf32>) permutation = [0, 2, 3, 1]
  %o = tensor.empty() : tensor<1x4x4x3xi8>
  %q = linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%t : tensor<1x4x4x3xf32>) outs(%o : tensor<1x4x4x3xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %i = arith.fptosi %d : f32 to i8
    linalg.yield %i : i8
  } -> tensor<1x4x4x3xi8>
  return %q : tensor<1x4x4x3xi8>
}

// The mirror of the collapse above, and the one a classifier needs: the flatten
// in front of it sits between a convolution's tail and the quantization of its
// result, so the quantization never gets next to the convolution and the layer
// stays in software. Moving it back across the reshape lets it fuse with the
// dequantize, which is the form --convert-linalg-to-gemmlir folds into the
// accelerator call. On the second CNN this was the difference between two of
// three convolutions reaching the accelerator and all three -- 8.75 ms to 0.95.
// CHECK-LABEL: func.func @quantize_before_the_flatten
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x2x2x2xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x2x2xi8>)
// CHECK:         tensor.collapse_shape %[[Q]] {{\[}}[0], [1, 2, 3]] {{.*}} into tensor<1x8xi8>
// CHECK-NOT:     linalg.generic
#id4c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#id2b = affine_map<(d0, d1) -> (d0, d1)>
func.func @quantize_before_the_flatten(%x: tensor<1x2x2x2xf32>) -> tensor<1x8xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %c = tensor.collapse_shape %x [[0], [1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<1x8xf32>
  %o = tensor.empty() : tensor<1x8xi8>
  %q = linalg.generic {indexing_maps = [#id2b, #id2b], iterator_types = ["parallel","parallel"]}
    ins(%c : tensor<1x8xf32>) outs(%o : tensor<1x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %i = arith.fptosi %d : f32 to i8
    linalg.yield %i : i8
  } -> tensor<1x8xi8>
  return %q : tensor<1x8xi8>
}

// Fusion copies the producer into the consumer, so a producer with more than
// one consumer gets computed more than once. A residual block is exactly that
// shape -- the block's input feeds both the first convolution and the shortcut
// -- and fusing across it recomputed a whole layer's tail *and* left two
// convolutions inside a four-operand `linalg.generic` that the accelerator
// matcher could not read, costing two of the three convolution folds.
//
// The stock pass's default control function refuses this already; this one is
// written from scratch to keep a quantization out of a contraction, and the
// condition had to be carried over with it. Both prefixes therefore check the
// same thing here -- the point of the test is that replacing the control
// function did not silently drop it.
// CHECK-LABEL: func.func @shared_producer_stays
// CHECK:         %[[P:.*]] = linalg.generic
// CHECK:           arith.mulf
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[P]] : tensor<8x8xf32>)
// CHECK:           arith.negf
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[P]] : tensor<8x8xf32>)
// CHECK:           arith.addf

// STOCK-LABEL: func.func @shared_producer_stays
// STOCK:         %[[P:.*]] = linalg.generic
// STOCK:           arith.mulf
// STOCK:         linalg.generic
// STOCK-SAME:      ins(%[[P]] : tensor<8x8xf32>)
// STOCK:         linalg.generic
// STOCK-SAME:      ins(%[[P]] : tensor<8x8xf32>)
func.func @shared_producer_stays(%a: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %two = arith.constant 2.0 : f32
  %e = tensor.empty() : tensor<8x8xf32>
  %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%a : tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %m = arith.mulf %x, %two : f32
    linalg.yield %m : f32
  } -> tensor<8x8xf32>
  %n = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%p : tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %g = arith.negf %x : f32
    linalg.yield %g : f32
  } -> tensor<8x8xf32>
  %s = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%p : tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %g = arith.addf %x, %two : f32
    linalg.yield %g : f32
  } -> tensor<8x8xf32>
  return %n, %s : tensor<8x8xf32>, tensor<8x8xf32>
}

// An i8 activation is where a layer ends: it is what the accelerator writes and
// what the next layer reads. Fusing it into the dequantization that follows
// puts the widening *inside* the producer, so the producing convolution's tail
// no longer ends in a requantization and --convert-linalg-to-gemmlir cannot
// fold it -- the convolution stays a scalar loop to save one pass over the
// activation. That is the shape --share-branch-quantization leaves at a
// residual block's branch point.
// CHECK-LABEL: func.func @quantized_activation_is_a_boundary
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK:           arith.trunci
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[Q]] : tensor<8xi8>)
// CHECK:           arith.sitofp

// The stock pass widens it back.
// STOCK-LABEL: func.func @quantized_activation_is_a_boundary
// STOCK-NOT:     tensor<8xi8>
// STOCK:         arith.sitofp
func.func @quantized_activation_is_a_boundary(%x: tensor<8xi32>) -> tensor<8xf32> {
  %s = arith.constant 2.0 : f32
  %e8 = tensor.empty() : tensor<8xi8>
  %ef = tensor.empty() : tensor<8xf32>
  %q = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
    ins(%x : tensor<8xi32>) outs(%e8 : tensor<8xi8>) {
  ^bb0(%in: i32, %o: i8):
    %t = arith.trunci %in : i32 to i8
    linalg.yield %t : i8
  } -> tensor<8xi8>
  %d = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
    ins(%q : tensor<8xi8>) outs(%ef : tensor<8xf32>) {
  ^bb0(%in: i8, %o: f32):
    %w = arith.sitofp %in : i8 to f32
    %m = arith.mulf %w, %s : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  return %d : tensor<8xf32>
}

// The same boundary, written the way a **transformer** writes it. Asking whether
// the consumer's *result* is f32 is the same question for a convolution network,
// where the dequantization is a tail of its own. It is not the same question
// here: the whole of a GELU and the next layer's requantization are one generic
// that reads an i8, widens it, and yields an i8 -- the result type says f32
// nowhere, so the widening went in anyway and twelve of a ViT's matmuls kept
// their i32 accumulator and a 17x768 f32 pass over it. **468.55 -> 398.69 ms**,
// at exactly the same 0.0553 relative L2.
//
// What decides is what the body does with the operand, not the type it ends at.
// CHECK-LABEL: func.func @an_i8_boundary_a_transformer_writes
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK:           arith.trunci
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[Q]] : tensor<8xi8>)
// CHECK:           arith.sitofp

// The stock pass fuses the two into one and the requantization stops being a
// tail the accelerator can end in. (The i8 type itself stays visible here --
// this function returns one.)
// STOCK-LABEL: func.func @an_i8_boundary_a_transformer_writes
// STOCK:         linalg.generic
// STOCK-NOT:     linalg.generic
func.func @an_i8_boundary_a_transformer_writes(%x: tensor<8xi32>) -> tensor<8xi8> {
  %s = arith.constant 2.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e8 = tensor.empty() : tensor<8xi8>
  %q = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
    ins(%x : tensor<8xi32>) outs(%e8 : tensor<8xi8>) {
  ^bb0(%in: i32, %o: i8):
    %t = arith.trunci %in : i32 to i8
    linalg.yield %t : i8
  } -> tensor<8xi8>
  // reads the i8, widens it, does something the mvout cannot, and quantizes
  // again -- an i8 in and an i8 out
  %g = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
    ins(%q : tensor<8xi8>) outs(%e8 : tensor<8xi8>) {
  ^bb0(%in: i8, %o: i8):
    %w = arith.sitofp %in : i8 to f32
    %er = math.erf %w : f32
    %m = arith.mulf %er, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<8xi8>
  return %g : tensor<8xi8>
}

// A requantization whose scales cancel, over a value that is already i8, is a
// copy. --share-branch-quantization quantizes a branching activation once and
// hands the other consumers the dequantization of it; where that other consumer
// is a second convolution reading the *same* tensor -- a ResNet stage
// transition, whose 1x1 projection and 3x3 convolution both read the block's
// input -- the calibration measured the same range for both, so it quantizes
// again at the scale it was just dequantized at. On ResNet-20 that was 24576 of
// the 27658 elements left.
//
// The ratio has to be accumulated as a fraction, not divided as the chain is
// walked: from the bottom, a scale of c over c comes out as (1/c)*c, which for
// most c is not 1. Getting that wrong left one of the two in place and looked
// like the pattern simply did not match.
// CHECK-LABEL: func.func @requantize_by_one
// CHECK-NOT:     linalg.generic
// CHECK:         return %arg0
// STOCK-LABEL: func.func @requantize_by_one
func.func @requantize_by_one(%x: tensor<8xi8>) -> tensor<8xi8> {
  %s = arith.constant 0.0272007 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<8xi8>
  %r = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
    ins(%x : tensor<8xi8>) outs(%e : tensor<8xi8>) {
  ^bb0(%in: i8, %o: i8):
    %w = arith.sitofp %in : i8 to f32
    %m = arith.mulf %w, %s : f32
    %d = arith.divf %m, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  } -> tensor<8xi8>
  return %r : tensor<8xi8>
}

// A real change of scale is a real operation.
// CHECK-LABEL: func.func @requantize_by_two
// CHECK:         linalg.generic
// CHECK:           arith.trunci
func.func @requantize_by_two(%x: tensor<8xi8>) -> tensor<8xi8> {
  %s = arith.constant 0.04 : f32
  %t2 = arith.constant 0.02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<8xi8>
  %r = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
    ins(%x : tensor<8xi8>) outs(%e : tensor<8xi8>) {
  ^bb0(%in: i8, %o: i8):
    %w = arith.sitofp %in : i8 to f32
    %m = arith.mulf %w, %s : f32
    %d = arith.divf %m, %t2 : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  } -> tensor<8xi8>
  return %r : tensor<8xi8>
}

// A clamp to [0, 127] is a relu, not a copy.
// CHECK-LABEL: func.func @requantize_with_relu
// CHECK:         linalg.generic
// CHECK:           arith.trunci
func.func @requantize_with_relu(%x: tensor<8xi8>) -> tensor<8xi8> {
  %s = arith.constant 0.0272007 : f32
  %lo = arith.constant 0 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<8xi8>
  %r = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
    ins(%x : tensor<8xi8>) outs(%e : tensor<8xi8>) {
  ^bb0(%in: i8, %o: i8):
    %w = arith.sitofp %in : i8 to f32
    %m = arith.mulf %w, %s : f32
    %d = arith.divf %m, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  } -> tensor<8xi8>
  return %r : tensor<8xi8>
}

// torch-mlir lowers a bounded activation -- ReLU6, Hardtanh -- by putting each
// bound in a 0-D tensor and broadcasting it, so what reaches the requantization
// is not `max(x, 0)` against a constant but against a block argument. Nothing
// that reads a body can see through that: the relu matcher looks for a zero and
// finds an argument, and a whole MobileNet block's convolutions stayed scalar
// loops because of it. The one value an operand carries everywhere belongs in
// the body.
// CHECK-LABEL: func.func @uniform_operand
// CHECK:         %[[K:.*]] = arith.constant 6.000000e+00 : f32
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<8xf32>)
// CHECK:           arith.minimumf %{{.*}}, %[[K]]
func.func @uniform_operand(%x: tensor<8xf32>) -> tensor<8xf32> {
  %six = arith.constant 6.000000e+00 : f32
  %s = tensor.empty() : tensor<f32>
  %b = linalg.fill ins(%six : f32) outs(%s : tensor<f32>) -> tensor<f32>
  %e = tensor.empty() : tensor<8xf32>
  %r = linalg.generic {indexing_maps = [#id1, affine_map<(d0) -> ()>, #id1],
                       iterator_types = ["parallel"]}
    ins(%x, %b : tensor<8xf32>, tensor<f32>) outs(%e : tensor<8xf32>) {
  ^bb0(%in: f32, %k: f32, %o: f32):
    %m = arith.minimumf %in, %k : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// `--share-branch-quantization` gives the branches of a split one scale, so an
// activation feeding two of them is dequantized to f32 once and quantized
// straight back -- at the *same* scale -- once per branch. Fusion will not
// merge the halves because the dequantization has more than one consumer, and
// recomputing a producer per branch is exactly what that rule exists to
// prevent; but this pair disappears, so there is nothing to recompute. On `atr`
// it is three passes of 9216 elements, a third of what the model had left in
// software.
//
// Exact, not approximate: `x * s` has the same significand as `x` for any `x`
// an i8 can hold, so `(x * s) / s` is within a relative 1e-7 of `x` and
// `roundeven` returns it, and the clamp cannot fire on a value that came out of
// an i8.
// CHECK-LABEL: func.func @round_trip_at_one_scale
// CHECK-NOT:     linalg.generic
// CHECK:         return %arg0, %arg0
#idr = affine_map<(d0, d1) -> (d0, d1)>
func.func @round_trip_at_one_scale(%x: tensor<16x32xi8>)
    -> (tensor<16x32xi8>, tensor<16x32xi8>) {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<16x32xf32>
  %d = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<16x32xi8>) outs(%e : tensor<16x32xf32>) {
  ^bb0(%v: i8, %o: f32):
    %f = arith.sitofp %v : i8 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<16x32xf32>
  %e1 = tensor.empty() : tensor<16x32xi8>
  %a = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%d : tensor<16x32xf32>) outs(%e1 : tensor<16x32xi8>) {
  ^bb0(%v: f32, %o: i8):
    %q = arith.divf %v, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<16x32xi8>
  %e2 = tensor.empty() : tensor<16x32xi8>
  %b = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%d : tensor<16x32xf32>) outs(%e2 : tensor<16x32xi8>) {
  ^bb0(%v: f32, %o: i8):
    %q = arith.divf %v, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<16x32xi8>
  return %a, %b : tensor<16x32xi8>, tensor<16x32xi8>
}

// A different scale on the way back is a real requantization and stays.
// CHECK-LABEL: func.func @round_trip_at_another_scale
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%{{.*}} : tensor<16x32xi8>)
func.func @round_trip_at_another_scale(%x: tensor<16x32xi8>) -> tensor<16x32xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %t2 = arith.constant 4.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<16x32xf32>
  %d = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<16x32xi8>) outs(%e : tensor<16x32xf32>) {
  ^bb0(%v: i8, %o: f32):
    %f = arith.sitofp %v : i8 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<16x32xf32>
  %e1 = tensor.empty() : tensor<16x32xi8>
  %a = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%d : tensor<16x32xf32>) outs(%e1 : tensor<16x32xi8>) {
  ^bb0(%v: f32, %o: i8):
    %q = arith.divf %v, %t2 : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<16x32xi8>
  return %a : tensor<16x32xi8>
}

// A relu between the two is not a copy either.
// CHECK-LABEL: func.func @round_trip_with_a_relu
// CHECK:         arith.maximumf
func.func @round_trip_with_a_relu(%x: tensor<16x32xi8>) -> tensor<16x32xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %z = arith.constant 0.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<16x32xf32>
  %d = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<16x32xi8>) outs(%e : tensor<16x32xf32>) {
  ^bb0(%v: i8, %o: f32):
    %f = arith.sitofp %v : i8 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<16x32xf32>
  %e1 = tensor.empty() : tensor<16x32xi8>
  %a = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%d : tensor<16x32xf32>) outs(%e1 : tensor<16x32xi8>) {
  ^bb0(%v: f32, %o: i8):
    %p = arith.maximumf %v, %z : f32
    %q = arith.divf %p, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<16x32xi8>
  return %a : tensor<16x32xi8>
}

// The pair may have a slice between it: a grouped convolution dequantizes the
// joined activation once and takes one slice per group, and each of those is
// quantized back at the same scale. Cutting the i8 instead is the same values,
// and it leaves everything above the dequantization alone -- which matters,
// because moving the slices up there instead costs a convolution its fold
// (`grp` 19.2 -> 22.6 ms).
// CHECK-LABEL: func.func @round_trip_through_a_slice
// CHECK:         %[[A:.*]] = tensor.extract_slice %arg0[0, 0] [16, 16] [1, 1]
// CHECK-SAME:      tensor<16x32xi8> to tensor<16x16xi8>
// CHECK:         %[[B:.*]] = tensor.extract_slice %arg0[0, 16] [16, 16] [1, 1]
// CHECK-SAME:      tensor<16x32xi8> to tensor<16x16xi8>
// CHECK:         return %[[A]], %[[B]]
func.func @round_trip_through_a_slice(%x: tensor<16x32xi8>)
    -> (tensor<16x16xi8>, tensor<16x16xi8>) {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<16x32xf32>
  %d = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<16x32xi8>) outs(%e : tensor<16x32xf32>) {
  ^bb0(%v: i8, %o: f32):
    %f = arith.sitofp %v : i8 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<16x32xf32>
  %sa = tensor.extract_slice %d[0, 0] [16, 16] [1, 1] : tensor<16x32xf32> to tensor<16x16xf32>
  %sb = tensor.extract_slice %d[0, 16] [16, 16] [1, 1] : tensor<16x32xf32> to tensor<16x16xf32>
  %e1 = tensor.empty() : tensor<16x16xi8>
  %a = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%sa : tensor<16x16xf32>) outs(%e1 : tensor<16x16xi8>) {
  ^bb0(%v: f32, %o: i8):
    %q = arith.divf %v, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<16x16xi8>
  %e2 = tensor.empty() : tensor<16x16xi8>
  %b = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%sb : tensor<16x16xf32>) outs(%e2 : tensor<16x16xi8>) {
  ^bb0(%v: f32, %o: i8):
    %q = arith.divf %v, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<16x16xi8>
  return %a, %b : tensor<16x16xi8>, tensor<16x16xi8>
}

// A stride in the slice changes nothing: a slice picks elements, and the same
// elements come out whichever side of the conversion it is taken on.
// CHECK-LABEL: func.func @strided_slice_between
// CHECK:         %[[S:.*]] = tensor.extract_slice %arg0[0, 0] [16, 16] [1, 2]
// CHECK-SAME:      tensor<16x32xi8> to tensor<16x16xi8>
// CHECK:         return %[[S]]
func.func @strided_slice_between(%x: tensor<16x32xi8>) -> tensor<16x16xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<16x32xf32>
  %d = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%x : tensor<16x32xi8>) outs(%e : tensor<16x32xf32>) {
  ^bb0(%v: i8, %o: f32):
    %f = arith.sitofp %v : i8 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<16x32xf32>
  %sa = tensor.extract_slice %d[0, 0] [16, 16] [1, 2] : tensor<16x32xf32> to tensor<16x16xf32>
  %e1 = tensor.empty() : tensor<16x16xi8>
  %a = linalg.generic {indexing_maps = [#idr, #idr], iterator_types = ["parallel","parallel"]}
    ins(%sa : tensor<16x16xf32>) outs(%e1 : tensor<16x16xi8>) {
  ^bb0(%v: f32, %o: i8):
    %q = arith.divf %v, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<16x16xi8>
  return %a : tensor<16x16xi8>
}

// A model that ends in NCHW dequantizes its last accumulator in NHWC and then
// relayouts the result, and the relayout is a `linalg.transpose` -- a named
// operation, which the elementwise fusion does not fuse into. Two full passes
// over the activation where one would do: 9216 elements of f32 written and read
// back for nothing on `atr` and `atrn`.
//
// The fused form iterates the producer's space and writes permuted, so the
// reads stay in order and only the writes scatter.
// CHECK-LABEL: func.func @dequantize_into_the_relayout
// CHECK:         %[[F:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$IN:.*]], #[[$OUT:.*]]]
// CHECK-SAME:      ins(%arg0 : tensor<1x24x24x16xi32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x16x24x24xf32>)
// CHECK:         return %[[F]]
// CHECK-NOT:     linalg.transpose
#id4t = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @dequantize_into_the_relayout(%acc: tensor<1x24x24x16xi32>) -> tensor<1x16x24x24xf32> {
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<1x24x24x16xf32>
  %d = linalg.generic {indexing_maps = [#id4t, #id4t], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc : tensor<1x24x24x16xi32>) outs(%e : tensor<1x24x24x16xf32>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x24x24x16xf32>
  %init = tensor.empty() : tensor<1x16x24x24xf32>
  %t = linalg.transpose ins(%d : tensor<1x24x24x16xf32>) outs(%init : tensor<1x16x24x24xf32>) permutation = [0, 3, 1, 2]
  return %t : tensor<1x16x24x24xf32>
}

// A relayout whose result is used more than once is left alone. Fusing one in
// the middle takes the elementwise operation out of reach of everything that
// would otherwise have fused *it*, and on the model set that costs more than
// the pass it saves: `atr` went from 63340 scalar elements to 81772, `shf` and
// `shu` from 2826 to 6922.
// CHECK-LABEL: func.func @relayout_used_twice
// CHECK:         linalg.transpose
func.func @relayout_used_twice(%acc: tensor<1x24x24x16xi32>)
    -> (tensor<1x16x24x24xf32>, tensor<1x16x24x24xf32>) {
  %s = arith.constant 2.000000e-02 : f32
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x24x24x16xf32>
  %d = linalg.generic {indexing_maps = [#id4t, #id4t], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc : tensor<1x24x24x16xi32>) outs(%e : tensor<1x24x24x16xf32>) {
  ^bb0(%v: i32, %o: f32):
    %f = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x24x24x16xf32>
  %init = tensor.empty() : tensor<1x16x24x24xf32>
  %t = linalg.transpose ins(%d : tensor<1x24x24x16xf32>) outs(%init : tensor<1x16x24x24xf32>) permutation = [0, 3, 1, 2]
  %e2 = tensor.empty() : tensor<1x16x24x24xf32>
  %r = linalg.generic {indexing_maps = [#id4t, #id4t], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%t : tensor<1x16x24x24xf32>) outs(%e2 : tensor<1x16x24x24xf32>) {
  ^bb0(%v: f32, %o: f32):
    %p = arith.maximumf %v, %z : f32
    linalg.yield %p : f32
  } -> tensor<1x16x24x24xf32>
  return %t, %r : tensor<1x16x24x24xf32>, tensor<1x16x24x24xf32>
}

// A slice or a reshape of something uniform is uniform. A grouped convolution's
// accumulator is one zero fill sliced per group, and the slice is what reaches
// the tail -- so `gmin` was writing 8192 f32 zeros and reading every one of
// them back to add nothing.
// CHECK-LABEL: func.func @uniform_through_a_slice
// CHECK:         %[[G:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x16x16x8xi32>)
// CHECK-NOT:     linalg.fill
#id4u = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @uniform_through_a_slice(%acc: tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xf32> {
  %zero = arith.constant 0.0 : f32
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<1x16x16x32xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x16x16x32xf32>) -> tensor<1x16x16x32xf32>
  %sl = tensor.extract_slice %f[0, 0, 0, 0] [1, 16, 16, 8] [1, 1, 1, 1]
    : tensor<1x16x16x32xf32> to tensor<1x16x16x8xf32>
  %o = tensor.empty() : tensor<1x16x16x8xf32>
  %r = linalg.generic {indexing_maps = [#id4u, #id4u, #id4u], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc, %sl : tensor<1x16x16x8xi32>, tensor<1x16x16x8xf32>) outs(%o : tensor<1x16x16x8xf32>) {
  ^bb0(%v: i32, %b: f32, %out: f32):
    %fv = arith.sitofp %v : i32 to f32
    %m = arith.mulf %fv, %s : f32
    %a = arith.addf %m, %b : f32
    linalg.yield %a : f32
  } -> tensor<1x16x16x8xf32>
  return %r : tensor<1x16x16x8xf32>
}

// A slice of something that is not uniform stays an operand.
// CHECK-LABEL: func.func @slice_of_a_real_tensor
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0, %{{.*}} : tensor<1x16x16x8xi32>, tensor<1x16x16x8xf32>)
func.func @slice_of_a_real_tensor(%acc: tensor<1x16x16x8xi32>, %b: tensor<1x16x16x32xf32>)
    -> tensor<1x16x16x8xf32> {
  %s = arith.constant 2.000000e-02 : f32
  %sl = tensor.extract_slice %b[0, 0, 0, 0] [1, 16, 16, 8] [1, 1, 1, 1]
    : tensor<1x16x16x32xf32> to tensor<1x16x16x8xf32>
  %o = tensor.empty() : tensor<1x16x16x8xf32>
  %r = linalg.generic {indexing_maps = [#id4u, #id4u, #id4u], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc, %sl : tensor<1x16x16x8xi32>, tensor<1x16x16x8xf32>) outs(%o : tensor<1x16x16x8xf32>) {
  ^bb0(%v: i32, %bb: f32, %out: f32):
    %fv = arith.sitofp %v : i32 to f32
    %m = arith.mulf %fv, %s : f32
    %a = arith.addf %m, %bb : f32
    linalg.yield %a : f32
  } -> tensor<1x16x16x8xf32>
  return %r : tensor<1x16x16x8xf32>
}

// Distributing a relayout over a join was tried and **reverted**: it takes
// `gmin` from 24.3 to 26.4 ms although it removes 8192 scalar elements. Four
// permuting writes into eight channels of a thirty-two channel buffer are
// worse than one bulk transpose that reads and writes contiguously, and the
// element count cannot see that.
