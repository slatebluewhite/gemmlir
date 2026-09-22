// PyTorch hands over `linalg.conv_2d_nchw_fchw`, and the layout decides how much
// of the layer can be offloaded: from NHWC, MLIR's img2col contracts
// (positions x patch) * (patch x channels), which puts the bias in the columns
// -- the one form Gemmini's repeating_bias can take. From NCHW the contraction
// comes out the other way round and the whole tail of the layer stays in
// software.

// RUN: gemmlir-opt --conv-nchw-to-nhwc --canonicalize %s | FileCheck %s

#chan = affine_map<(n, c, h, w) -> (c)>
#nchw = affine_map<(n, c, h, w) -> (n, c, h, w)>

// Two layers joined by a relu. The transposes the rewrite puts around each
// convolution walk into the relu from both sides and compose to the identity,
// so what is left is one transpose on the way in and one on the way out -- the
// relu reads the first convolution's NHWC result directly. The filter is a
// constant and is permuted here rather than on every inference.
// CHECK-LABEL: func.func @two_layers
// CHECK:         %[[IN:.*]] = linalg.transpose ins(%arg0
// CHECK-SAME:      permutation = [0, 2, 3, 1]
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      ins(%[[IN]], %{{.*}} : tensor<1x6x6x2xf32>, tensor<3x3x2x2xf32>)
// CHECK:         linalg.generic
// CHECK:           arith.maximumf
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      -> tensor<1x2x2x2xf32>
// CHECK:         linalg.transpose
// CHECK-SAME:      permutation = [0, 3, 1, 2]
// CHECK-NEXT:    return
// CHECK-NOT:     conv_2d_nchw_fchw
func.func @two_layers(%in: tensor<1x2x6x6xf32>, %b1: tensor<2xf32>, %b2: tensor<2xf32>) -> tensor<1x2x2x2xf32> {
  %zero = arith.constant 0.0 : f32
  %f1 = arith.constant dense<1.0> : tensor<2x2x3x3xf32>
  %f2 = arith.constant dense<[[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                               [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.7, 1.8, 1.9]]],
                              [[[2.1, 2.2, 2.3], [2.4, 2.5, 2.6], [2.7, 2.8, 2.9]],
                               [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6], [3.7, 3.8, 3.9]]]]> : tensor<2x2x3x3xf32>
  %e1 = tensor.empty() : tensor<1x2x4x4xf32>
  %i1 = linalg.generic {indexing_maps = [#chan, #nchw], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%b1 : tensor<2xf32>) outs(%e1 : tensor<1x2x4x4xf32>) {
  ^bb0(%v: f32, %o: f32):
    linalg.yield %v : f32
  } -> tensor<1x2x4x4xf32>
  %c1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
     ins(%in, %f1 : tensor<1x2x6x6xf32>, tensor<2x2x3x3xf32>) outs(%i1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %r1 = linalg.generic {indexing_maps = [#nchw, #nchw], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c1 : tensor<1x2x4x4xf32>) outs(%e1 : tensor<1x2x4x4xf32>) {
  ^bb0(%v: f32, %o: f32):
    %m = arith.maximumf %v, %zero : f32
    linalg.yield %m : f32
  } -> tensor<1x2x4x4xf32>
  %e2 = tensor.empty() : tensor<1x2x2x2xf32>
  %i2 = linalg.generic {indexing_maps = [#chan, #nchw], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%b2 : tensor<2xf32>) outs(%e2 : tensor<1x2x2x2xf32>) {
  ^bb0(%v: f32, %o: f32):
    linalg.yield %v : f32
  } -> tensor<1x2x2x2xf32>
  %c2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
     ins(%r1, %f2 : tensor<1x2x4x4xf32>, tensor<2x2x3x3xf32>) outs(%i2 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %c2 : tensor<1x2x2x2xf32>
}

// A max-pool goes the same way, and its -inf fill is rewritten rather than
// transposed.
// CHECK-LABEL: func.func @pooling
// CHECK:         linalg.fill
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x2x2xf32>)
// CHECK:         linalg.pooling_nhwc_max
// CHECK-NOT:     pooling_nchw_max
func.func @pooling(%in: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %w = tensor.empty() : tensor<2x2xf32>
  %p = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%in, %w : tensor<1x2x4x4xf32>, tensor<2x2xf32>) outs(%f : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %p : tensor<1x2x2x2xf32>
}

// Sum pooling goes the same way. It is here because that is what a frontend
// lowers an average pool to -- the divide comes after it -- and
// --average-pool-to-contraction needs it in NHWC, where the spatial axes are next to
// each other and the channel is innermost, so the image becomes a matrix
// without moving anything.
// CHECK-LABEL: func.func @sum_pooling
// CHECK:         linalg.pooling_nhwc_sum
// CHECK-NOT:     pooling_nchw_sum
func.func @sum_pooling(%in: tensor<1x2x4x4xf32>) -> tensor<1x2x1x1xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x2x1x1xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
  %w = tensor.empty() : tensor<4x4xf32>
  %p = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : tensor<1x2x4x4xf32>, tensor<4x4xf32>) outs(%f : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
  return %p : tensor<1x2x1x1xf32>
}

// The filter's permutation is a compile-time evaluation: FCHW to HWCF is
// [2, 3, 1, 0], so filter[f][c][kh][kw] lands at [kh][kw][c][f].
// CHECK-LABEL: func.func @constant_filter_is_folded
// CHECK:         arith.constant dense<{{.*}}1.000000e+00, 5.000000e+00{{.*}}3.000000e+00, 7.000000e+00{{.*}}> : tensor<1x2x2x2xf32>
// CHECK-NOT:     permutation = [2, 3, 1, 0]
func.func @constant_filter_is_folded(%in: tensor<1x2x4x4xf32>, %init: tensor<1x2x4x2xf32>) -> tensor<1x2x4x2xf32> {
  // f=2, c=2, kh=1, kw=2
  %f = arith.constant dense<[[[[1.0, 2.0]], [[3.0, 4.0]]],
                             [[[5.0, 6.0]], [[7.0, 8.0]]]]> : tensor<2x2x1x2xf32>
  %c = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
     ins(%in, %f : tensor<1x2x4x4xf32>, tensor<2x2x1x2xf32>) outs(%init : tensor<1x2x4x2xf32>) -> tensor<1x2x4x2xf32>
  return %c : tensor<1x2x4x2xf32>
}

// A convolution block ends NHWC and the flatten before the classifier expects
// the frontend's NCHW order, so one transpose is left there with nothing to
// cancel against. It does not have to run: flattening the other order just
// permutes which weight row each activation meets, and the weights are
// constants. Here the pooled activation is 1x2x2x2 -- (c,h,w) order becomes
// (h,w,c) -- so weight row (c,h,w) moves to (h,w,c): rows 0..7 in the order
// 0, 4, 1, 5, 2, 6, 3, 7.
// CHECK-LABEL: func.func @transpose_moves_into_the_weights
// CHECK-NOT:     linalg.transpose
// CHECK:         %[[W:.*]] = arith.constant dense<{{\[}}[0.000000e+00], [4.000000e+00], [1.000000e+00], [5.000000e+00], [2.000000e+00], [6.000000e+00], [3.000000e+00], [7.000000e+00]]> : tensor<8x1xf32>
// CHECK:         %[[F:.*]] = tensor.collapse_shape %arg0 {{\[}}[0], [1, 2, 3]] {{.*}} into tensor<1x8xf32>
// CHECK:         linalg.matmul ins(%[[F]], %[[W]]
func.func @transpose_moves_into_the_weights(%pooled: tensor<1x2x2x2xf32>, %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %w = arith.constant dense<[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]> : tensor<8x1xf32>
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %t = linalg.transpose ins(%pooled : tensor<1x2x2x2xf32>) outs(%e : tensor<1x2x2x2xf32>) permutation = [0, 3, 1, 2]
  %c = tensor.collapse_shape %t [[0], [1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<1x8xf32>
  %r = linalg.matmul ins(%c, %w : tensor<1x8xf32>, tensor<8x1xf32>) outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %r : tensor<1x1xf32>
}

// With weights that are not constant the permutation would happen on every
// inference, moving K x N elements to save K. Left alone.
// CHECK-LABEL: func.func @dynamic_weights_stay
// CHECK:         linalg.transpose
// CHECK:         linalg.matmul
func.func @dynamic_weights_stay(%pooled: tensor<1x2x2x2xf32>, %w: tensor<8x1xf32>,
                                %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %t = linalg.transpose ins(%pooled : tensor<1x2x2x2xf32>) outs(%e : tensor<1x2x2x2xf32>) permutation = [0, 3, 1, 2]
  %c = tensor.collapse_shape %t [[0], [1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<1x8xf32>
  %r = linalg.matmul ins(%c, %w : tensor<1x8xf32>, tensor<8x1xf32>) outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %r : tensor<1x1xf32>
}

// Two patterns want this transpose: pushing it past the elementwise operation
// only moves it, while moving it into the weights removes it. The push gives
// way when the transpose is the flatten in front of a classifier whose weights
// are constant -- without that, a relu between the convolution and the flatten
// is enough to swallow the transpose and strand the layer in software, which is
// what a second CNN with a relu there turned up.
// CHECK-LABEL: func.func @the_weights_win
// CHECK-NOT:     linalg.transpose
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK:           arith.maximumf
// CHECK:         %[[F:.*]] = tensor.collapse_shape %[[R]]
// CHECK:         linalg.matmul ins(%[[F]]
#id4b = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @the_weights_win(%x: tensor<1x2x2x2xf32>, %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %zero = arith.constant 0.0 : f32
  %w = arith.constant dense<[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]> : tensor<8x1xf32>
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %r = linalg.generic {indexing_maps = [#id4b, #id4b], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%x : tensor<1x2x2x2xf32>) outs(%e : tensor<1x2x2x2xf32>) {
  ^bb0(%v: f32, %o: f32):
    %m = arith.maximumf %v, %zero : f32
    linalg.yield %m : f32
  } -> tensor<1x2x2x2xf32>
  %t = linalg.transpose ins(%r : tensor<1x2x2x2xf32>) outs(%e : tensor<1x2x2x2xf32>) permutation = [0, 3, 1, 2]
  %c = tensor.collapse_shape %t [[0], [1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<1x8xf32>
  %m = linalg.matmul ins(%c, %w : tensor<1x8xf32>, tensor<8x1xf32>) outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %m : tensor<1x1xf32>
}

// When the relu gets to the transpose first there is no `linalg.transpose` left
// to match, and the relu itself is what produces the frontend's layout. It can
// be rewritten to produce the convolution's instead -- every map composed with
// the inverse permutation, which makes the convolution's own read the identity
// -- and the classifier's weight rows permuted to match. Note the read map
// spells the batch dimension as a constant 0, which is how a frontend writes an
// axis of extent 1; the match has to allow it.
// CHECK-LABEL: func.func @permutation_inside_the_relu
// CHECK:         %[[W:.*]] = arith.constant dense<{{\[}}[0.000000e+00], [4.000000e+00], [1.000000e+00], [5.000000e+00], [2.000000e+00], [6.000000e+00], [3.000000e+00], [7.000000e+00]]> : tensor<8x1xf32>
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x2x2x2xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x2x2xf32>)
// CHECK:         %[[F:.*]] = tensor.collapse_shape %[[R]]
// CHECK:         linalg.matmul ins(%[[F]], %[[W]]
#nchwFrom = affine_map<(n, c, h, w) -> (0, h, w, c)>
#idNchw = affine_map<(n, c, h, w) -> (n, c, h, w)>
func.func @permutation_inside_the_relu(%conv: tensor<1x2x2x2xf32>, %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %zero = arith.constant 0.0 : f32
  %w = arith.constant dense<[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]> : tensor<8x1xf32>
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %r = linalg.generic {indexing_maps = [#nchwFrom, #idNchw], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%conv : tensor<1x2x2x2xf32>) outs(%e : tensor<1x2x2x2xf32>) {
  ^bb0(%v: f32, %o: f32):
    %m = arith.maximumf %v, %zero : f32
    linalg.yield %m : f32
  } -> tensor<1x2x2x2xf32>
  %c = tensor.collapse_shape %r [[0], [1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<1x8xf32>
  %m = linalg.matmul ins(%c, %w : tensor<1x8xf32>, tensor<8x1xf32>) outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %m : tensor<1x1xf32>
}

// Absorbing a transpose into an elementwise operation rewrites the operand's
// map and leaves the transpose that used to feed its *destination* behind --
// still a use of the convolution's result, so the result no longer looks
// single-use to anything downstream, and still a real transpose of the whole
// activation at run time. The destination only supplies a shape, so it can be a
// fresh `tensor.empty`.
// CHECK-LABEL: func.func @dead_destination
// CHECK:         %[[C:.*]] = linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:     linalg.transpose ins(%[[C]]
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[C]]
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x4x4xf32>)
func.func @dead_destination(%in: tensor<1x1x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = arith.constant dense<[[[[2.0, 4.0]]]]> : tensor<1x1x1x2xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x4x4x2xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32>
  %e2 = tensor.empty() : tensor<1x4x4x1xf32>
  %nhwc = linalg.transpose ins(%in : tensor<1x1x4x4xf32>) outs(%e2 : tensor<1x4x4x1xf32>) permutation = [0, 2, 3, 1]
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%nhwc, %w : tensor<1x4x4x1xf32>, tensor<1x1x1x2xf32>)
    outs(%f : tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32>
  %o = tensor.empty() : tensor<1x2x4x4xf32>
  %back = linalg.transpose ins(%c : tensor<1x4x4x2xf32>) outs(%o : tensor<1x2x4x4xf32>) permutation = [0, 3, 1, 2]
  %r = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(0,d2,d3,d1)>,
                                        affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x4x4x2xf32>) outs(%back : tensor<1x2x4x4xf32>) {
  ^bb0(%x: f32, %out: f32):
    %g = arith.cmpf ugt, %x, %zero : f32
    %s = arith.select %g, %x, %zero : f32
    linalg.yield %s : f32
  } -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// A generic that only reads a constant, at indices it computes from its own
// loop indices, is a constant. MLIR's populateConstantFoldLinalgOperations does
// not reach it: that folds an elementwise operation over constant *operands*,
// and this one has none -- it gathers, with linalg.index and a tensor.extract.
// It is how torch-mlir writes the kernel a transposed convolution needs, which
// is the forward kernel with its spatial axes reflected and its channels
// swapped. Left alone it is recomputed every inference, and the filter is not a
// compile-time constant, so --force-quantized-matmul has no range for it and
// quantizes it at the *activation's* scale.
//
// weights[f][c][kh][kw] with f,c in 0..1 and a 2x2 kernel, read as
// [c][f][1-kh][1-kw], is the reflection below.
// CHECK-LABEL: func.func @constant_gather
// CHECK:         arith.constant dense<{{\[\[\[}}[4.000000e+00, 3.000000e+00], [2.000000e+00, 1.000000e+00]{{\]}}, {{\[}}[8.000000e+00, 7.000000e+00], [6.000000e+00, 5.000000e+00]{{\]\]\]}}> : tensor<1x2x2x2xf32>
// CHECK-NOT:     linalg.index
func.func @constant_gather() -> tensor<1x2x2x2xf32> {
  %w = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]]>
       : tensor<2x1x2x2xf32>
  %c1 = arith.constant 1 : index
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %g = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    outs(%e : tensor<1x2x2x2xf32>) {
  ^bb0(%o: f32):
    %i0 = linalg.index 0 : index
    %i1 = linalg.index 1 : index
    %i2 = linalg.index 2 : index
    %i3 = linalg.index 3 : index
    %r2 = arith.subi %c1, %i2 : index
    %r3 = arith.subi %c1, %i3 : index
    %v = tensor.extract %w[%i1, %i0, %r2, %r3] : tensor<2x1x2x2xf32>
    linalg.yield %v : f32
  } -> tensor<1x2x2x2xf32>
  return %g : tensor<1x2x2x2xf32>
}
