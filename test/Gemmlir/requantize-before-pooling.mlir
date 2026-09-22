// `max` commutes with any non-decreasing function, so a requantization sitting
// after a max-pool can go before it. That matters because it is what gives the
// convolution an i8 result to fold into: a quantized network as a frontend
// writes it pools in f32 and requantizes afterwards, which leaves the layer's
// whole tail -- dequantize, bias, activation -- with nothing to fold into, and
// the convolution stays in software. On the CNN that was the difference between
// one convolution reaching the accelerator and both.

// RUN: gemmlir-opt --requantize-before-pooling %s | FileCheck %s

#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>

// The requantization moves to the pool's input and the pool runs on i8, whose
// accumulator starts at the type's own minimum rather than -inf.
// CHECK-LABEL: func.func @moves_before
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x4x4x2xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x4x4x2xi8>)
// CHECK:         %[[F:.*]] = linalg.fill ins(%c-128{{.*}} : i8)
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      ins(%[[Q]], %{{.*}} : tensor<1x4x4x2xi8>, tensor<2x2xf32>)
// CHECK-SAME:      outs(%[[F]] : tensor<1x2x2x2xi8>)
func.func @moves_before(%in: tensor<1x4x4x2xf32>) -> tensor<1x2x2x2xi8> {
  %ninf = arith.constant 0xFF800000 : f32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %w = tensor.empty() : tensor<2x2xf32>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%in, %w : tensor<1x4x4x2xf32>, tensor<2x2xf32>) outs(%f : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %o = tensor.empty() : tensor<1x2x2x2xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%p : tensor<1x2x2x2xf32>) outs(%o : tensor<1x2x2x2xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %i, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x2x2x2xi8>
  return %q : tensor<1x2x2x2xi8>
}

// The frontend flattens the pooled activation before the classifier and the
// layout rewrite leaves a transpose in front of that, so the requantization
// reads the pool through a chain of shape-only operations. They are replayed
// after the moved pool rather than matched away.
// CHECK-LABEL: func.func @through_a_reshape
// CHECK:         %[[Q2:.*]] = linalg.generic
// CHECK-SAME:      outs(%{{.*}} : tensor<1x4x4x2xi8>)
// CHECK:         %[[P2:.*]] = linalg.pooling_nhwc_max
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x2x2xi8>)
// CHECK:         %[[T:.*]] = linalg.transpose ins(%[[P2]]
// CHECK-SAME:      permutation = [0, 3, 1, 2]
// CHECK:         tensor.collapse_shape %[[T]]
func.func @through_a_reshape(%in: tensor<1x4x4x2xf32>) -> tensor<1x8xi8> {
  %ninf = arith.constant 0xFF800000 : f32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %w = tensor.empty() : tensor<2x2xf32>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%in, %w : tensor<1x4x4x2xf32>, tensor<2x2xf32>) outs(%f : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %te = tensor.empty() : tensor<1x2x2x2xf32>
  %t = linalg.transpose ins(%p : tensor<1x2x2x2xf32>) outs(%te : tensor<1x2x2x2xf32>) permutation = [0, 3, 1, 2]
  %c = tensor.collapse_shape %t [[0], [1, 2, 3]] : tensor<1x2x2x2xf32> into tensor<1x8xf32>
  %o = tensor.empty() : tensor<1x8xi8>
  %q = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel","parallel"]}
    ins(%c : tensor<1x8xf32>) outs(%o : tensor<1x8xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %i, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %tr = arith.trunci %b : i32 to i8
    linalg.yield %tr : i8
  } -> tensor<1x8xi8>
  return %q : tensor<1x8xi8>
}

// Scaling by a *negative* constant reverses the order, so the maximum of the
// window is no longer the maximum of what comes out. Left alone.
// CHECK-LABEL: func.func @negative_scale_stays
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x2x2xf32>)
// CHECK:         linalg.generic
func.func @negative_scale_stays(%in: tensor<1x4x4x2xf32>) -> tensor<1x2x2x2xi8> {
  %ninf = arith.constant 0xFF800000 : f32
  %s = arith.constant -2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %w = tensor.empty() : tensor<2x2xf32>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%in, %w : tensor<1x4x4x2xf32>, tensor<2x2xf32>) outs(%f : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %o = tensor.empty() : tensor<1x2x2x2xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%p : tensor<1x2x2x2xf32>) outs(%o : tensor<1x2x2x2xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.mulf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %i, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x2x2x2xi8>
  return %q : tensor<1x2x2x2xi8>
}

// A truncation with no clamp in front of it wraps, and wrapping is not
// monotonic. Left alone.
// CHECK-LABEL: func.func @unclamped_truncation_stays
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x2x2xf32>)
// CHECK:         linalg.generic
func.func @unclamped_truncation_stays(%in: tensor<1x4x4x2xf32>) -> tensor<1x2x2x2xi8> {
  %ninf = arith.constant 0xFF800000 : f32
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %w = tensor.empty() : tensor<2x2xf32>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%in, %w : tensor<1x4x4x2xf32>, tensor<2x2xf32>) outs(%f : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %o = tensor.empty() : tensor<1x2x2x2xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%p : tensor<1x2x2x2xf32>) outs(%o : tensor<1x2x2x2xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %i = arith.fptosi %d : f32 to i32
    %t = arith.trunci %i : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x2x2x2xi8>
  return %q : tensor<1x2x2x2xi8>
}

// The same move on an NCHW pool. `max` commutes with a non-decreasing function
// whatever order the axes are written in, so nothing here is about layout --
// and this is the spelling a model whose convolution became an im2col matmul is
// left with, because `--conv-nchw-to-nhwc` is not in the quantized pipeline and
// nothing else turns it over. `cnn_full` and `cnn_i2c` were pooling 3136 f32
// elements and quantizing 784 more afterwards; both now pool on i8, and the
// matmul's tail ends in a requantization instead of an f32 buffer.
// CHECK-LABEL: func.func @nchw_pool_moves
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x4x4xi8>)
// CHECK:         linalg.fill ins(%c-128_i8
// CHECK:         linalg.pooling_nchw_max
// CHECK-SAME:      ins(%[[Q]], %{{.*}} : tensor<1x2x4x4xi8>, tensor<2x2xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x2x2x2xi8>)
// CHECK-NOT:     linalg.generic
// CHECK:         return
func.func @nchw_pool_moves(%in: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xi8> {
  %ninf = arith.constant 0xFF800000 : f32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %w = tensor.empty() : tensor<2x2xf32>
  %p = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%in, %w : tensor<1x2x4x4xf32>, tensor<2x2xf32>) outs(%f : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %o = tensor.empty() : tensor<1x2x2x2xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%p : tensor<1x2x2x2xf32>) outs(%o : tensor<1x2x2x2xi8>) {
  ^bb0(%v: f32, %out: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %i, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x2x2x2xi8>
  return %q : tensor<1x2x2x2xi8>
}

// GoogLeNet's inception pool feeds the requantization *and*, through a pad, the
// next module's pool -- and once that one has been moved, the second reader is
// the very same requantization. `hasOneUse` refused both of the two pools that
// mattered, which are exactly the ones whose input is a concatenation of four
// dequantized branches.
//
// A second reader is handed the **inverse**: one multiply by the scale that
// moved. `quantize(dequantize(q))` is `q` for a byte, so a reader that
// quantizes again sees no difference.

// CHECK-LABEL: func.func @a_second_reader_gets_the_inverse
// The pool is i8 now...
// CHECK:       %[[P:.*]] = linalg.pooling_nhwc_max
// CHECK-SAME:  tensor<1x6x6x8xi8>
// ...and what else read it in f32 gets the scale multiplied back in:
// CHECK:       linalg.generic
// CHECK-SAME:  ins(%[[P]]
// CHECK:         arith.sitofp
// CHECK:         arith.mulf
func.func @a_second_reader_gets_the_inverse(%x: tensor<1x8x8x8xf32>, %w: tensor<3x3xf32>)
    -> (tensor<1x6x6x8xi8>, tensor<1x4x4x8xf32>) {
  %cst = arith.constant 2.500000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %ninf = arith.constant -3.40282347E+38 : f32
  %e = tensor.empty() : tensor<1x6x6x8xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%x, %w : tensor<1x8x8x8xf32>, tensor<3x3xf32>)
      outs(%f : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
  %o = tensor.empty() : tensor<1x6x6x8xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%p : tensor<1x6x6x8xf32>) outs(%o : tensor<1x6x6x8xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %cst : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %i, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x6x6x8xi8>
  %second = tensor.empty() : tensor<1x4x4x8xf32>
  %s = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%p, %w : tensor<1x6x6x8xf32>, tensor<3x3xf32>)
      outs(%second : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  return %q, %s : tensor<1x6x6x8xi8>, tensor<1x4x4x8xf32>
}

// A second reader that is neither a max-pool nor the same requantization would
// see the rounded value where it used to see the exact one, so the pool stays
// in f32.

// CHECK-LABEL: func.func @b_a_reader_with_no_inverse
// CHECK:       linalg.pooling_nhwc_max
// CHECK-SAME:  tensor<1x6x6x8xf32>
func.func @b_a_reader_with_no_inverse(%x: tensor<1x8x8x8xf32>, %w: tensor<3x3xf32>)
    -> (tensor<1x6x6x8xi8>, tensor<1x6x6x8xf32>) {
  %cst = arith.constant 2.500000e-02 : f32
  %two = arith.constant 2.000000e+00 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %ninf = arith.constant -3.40282347E+38 : f32
  %e = tensor.empty() : tensor<1x6x6x8xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%x, %w : tensor<1x8x8x8xf32>, tensor<3x3xf32>)
      outs(%f : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
  %o = tensor.empty() : tensor<1x6x6x8xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%p : tensor<1x6x6x8xf32>) outs(%o : tensor<1x6x6x8xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %cst : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %a = arith.maxsi %i, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %t = arith.trunci %b : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x6x6x8xi8>
  %e2 = tensor.empty() : tensor<1x6x6x8xf32>
  %s = linalg.generic {indexing_maps = [#id4, #id4],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%p : tensor<1x6x6x8xf32>) outs(%e2 : tensor<1x6x6x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.mulf %in, %two : f32
    linalg.yield %m : f32
  } -> tensor<1x6x6x8xf32>
  return %q, %s : tensor<1x6x6x8xi8>, tensor<1x6x6x8xf32>
}
