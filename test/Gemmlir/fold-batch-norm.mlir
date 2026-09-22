// A batch norm in evaluation mode is a per-output-channel affine, and every
// term of it is a constant. Nothing downstream can take it -- tiled_conv_auto
// scales its accumulator by one number, not one per channel -- so a convolution
// followed by one stays a scalar loop. All of it goes into the weights.

// RUN: gemmlir-opt --fold-batch-norm --canonicalize %s | FileCheck %s

#img = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d1)>

// Channel 0: gamma 3, var 0.75, eps 0.25, so a = 3 / sqrt(1) = 3; mean 2 and
// beta 1 give b = 1 - 2*3 = -5. Channel 1: gamma 2 over sqrt(4) = 1, and
// b = -1 - 0.5*1 = -1.5.
//
// The filter is [2, 4], so it becomes [2*3, 4*1] = [6, 4], and the 0.5 the
// convolution was accumulating onto goes through the same affine:
// [0.5*3 - 5, 0.5*1 - 1.5] = [-3.5, -1].
// CHECK-LABEL: func.func @batch_norm
// CHECK-DAG:     %[[W:.*]] = arith.constant dense<{{\[}}{{\[}}{{\[}}[6.000000e+00]{{\]}}{{\]}}, {{\[}}{{\[}}[4.000000e+00]{{\]}}{{\]}}{{\]}}> : tensor<2x1x1x1xf32>
// CHECK-DAG:     %[[B:.*]] = arith.constant dense<[-3.500000e+00, -1.000000e+00]> : tensor<2xf32>
// CHECK:         %[[INIT:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[B]] : tensor<2xf32>)
// CHECK:         linalg.conv_2d_nchw_fchw
// CHECK-SAME:      ins(%{{.*}}, %[[W]] :
// CHECK-SAME:      outs(%[[INIT]]
// CHECK-NOT:     math.rsqrt
func.func @batch_norm(%in: tensor<1x1x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = arith.constant dense<[[[[2.0]]], [[[4.0]]]]> : tensor<2x1x1x1xf32>
  %gamma = arith.constant dense<[3.0, 2.0]> : tensor<2xf32>
  %beta = arith.constant dense<[1.0, -1.0]> : tensor<2xf32>
  %mean = arith.constant dense<[2.0, 0.5]> : tensor<2xf32>
  %var = arith.constant dense<[0.75, 3.75]> : tensor<2xf32>
  %eps = arith.constant 2.500000e-01 : f32
  %half = arith.constant 5.000000e-01 : f32
  %e = tensor.empty() : tensor<1x2x4x4xf32>
  %f = linalg.fill ins(%half : f32) outs(%e : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %c = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
    outs(%f : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %bn = linalg.generic {indexing_maps = [#img, #chan, #chan, #chan, #chan, #img],
                        iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c, %gamma, %beta, %mean, %var : tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
    outs(%c : tensor<1x2x4x4xf32>) {
  ^bb0(%x: f32, %g: f32, %b: f32, %m: f32, %v: f32, %o: f32):
    %ve = arith.addf %v, %eps : f32
    %inv = math.rsqrt %ve : f32
    %d = arith.subf %x, %m : f32
    %s = arith.mulf %d, %inv : f32
    %sg = arith.mulf %s, %g : f32
    %r = arith.addf %sg, %b : f32
    linalg.yield %r : f32
  } -> tensor<1x2x4x4xf32>
  return %bn : tensor<1x2x4x4xf32>
}

// A plain per-channel scale is the same shape with b = 0, and a classifier's
// weights are K x N, so the channel is the trailing one.
// CHECK-LABEL: func.func @scale_after_matmul
// CHECK:         arith.constant dense<{{\[}}[2.000000e+00, 6.000000e+00], [4.000000e+00, 1.200000e+01]{{\]}}> : tensor<2x2xf32>
// CHECK:         linalg.matmul
// CHECK-NOT:     linalg.generic
func.func @scale_after_matmul(%in: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %w = arith.constant dense<[[2.0, 2.0], [4.0, 4.0]]> : tensor<2x2xf32>
  %s = arith.constant dense<[1.0, 3.0]> : tensor<2xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<4x2xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
  %m = linalg.matmul ins(%in, %w : tensor<4x2xf32>, tensor<2x2xf32>)
       outs(%f : tensor<4x2xf32>) -> tensor<4x2xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>],
                       iterator_types = ["parallel","parallel"]}
    ins(%m, %s : tensor<4x2xf32>, tensor<2xf32>) outs(%e : tensor<4x2xf32>) {
  ^bb0(%x: f32, %k: f32, %o: f32):
    %p = arith.mulf %x, %k : f32
    linalg.yield %p : f32
  } -> tensor<4x2xf32>
  return %r : tensor<4x2xf32>
}

// Squaring the activation is not affine in it, and there is no weight that
// makes it so.
// CHECK-LABEL: func.func @not_affine
// CHECK:         linalg.conv_2d_nchw_fchw
// CHECK:         linalg.generic
// CHECK:           arith.mulf
func.func @not_affine(%in: tensor<1x1x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = arith.constant dense<[[[[2.0]]], [[[4.0]]]]> : tensor<2x1x1x1xf32>
  %s = arith.constant dense<[1.0, 3.0]> : tensor<2xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x2x4x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %c = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
    outs(%f : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %r = linalg.generic {indexing_maps = [#img, #chan, #img],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c, %s : tensor<1x2x4x4xf32>, tensor<2xf32>) outs(%e : tensor<1x2x4x4xf32>) {
  ^bb0(%x: f32, %k: f32, %o: f32):
    %p = arith.mulf %x, %x : f32
    %q = arith.mulf %p, %k : f32
    linalg.yield %q : f32
  } -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// A scale that is only known at run time cannot go into a compile-time
// constant.
// CHECK-LABEL: func.func @runtime_scale
// CHECK:         linalg.conv_2d_nchw_fchw
// CHECK:         linalg.generic
// CHECK:           arith.mulf
func.func @runtime_scale(%in: tensor<1x1x4x4xf32>, %s: tensor<2xf32>) -> tensor<1x2x4x4xf32> {
  %w = arith.constant dense<[[[[2.0]]], [[[4.0]]]]> : tensor<2x1x1x1xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x2x4x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %c = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
    outs(%f : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %r = linalg.generic {indexing_maps = [#img, #chan, #img],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c, %s : tensor<1x2x4x4xf32>, tensor<2xf32>) outs(%e : tensor<1x2x4x4xf32>) {
  ^bb0(%x: f32, %k: f32, %o: f32):
    %p = arith.mulf %x, %k : f32
    linalg.yield %p : f32
  } -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// The convolution's result is wanted somewhere else as well, so it cannot be
// rewritten out from under that use.
// CHECK-LABEL: func.func @result_used_twice
// CHECK:         linalg.conv_2d_nchw_fchw
// CHECK:         linalg.generic
// CHECK:           arith.mulf
func.func @result_used_twice(%in: tensor<1x1x4x4xf32>)
    -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
  %w = arith.constant dense<[[[[2.0]]], [[[4.0]]]]> : tensor<2x1x1x1xf32>
  %s = arith.constant dense<[1.0, 3.0]> : tensor<2xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x2x4x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %c = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : tensor<1x1x4x4xf32>, tensor<2x1x1x1xf32>)
    outs(%f : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %r = linalg.generic {indexing_maps = [#img, #chan, #img],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c, %s : tensor<1x2x4x4xf32>, tensor<2xf32>) outs(%e : tensor<1x2x4x4xf32>) {
  ^bb0(%x: f32, %k: f32, %o: f32):
    %p = arith.mulf %x, %k : f32
    linalg.yield %p : f32
  } -> tensor<1x2x4x4xf32>
  return %r, %c : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
}

// A scale of one is not a scale. A plain bias after a contraction is already
// something the pipeline reads -- it becomes the accelerator call's D operand
// -- so rewriting the weights for it would churn the IR to no purpose.
// CHECK-LABEL: func.func @bias_only
// CHECK:         linalg.matmul
// CHECK:         linalg.generic
// CHECK:           arith.addf
func.func @bias_only(%in: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %w = arith.constant dense<[[2.0, 2.0], [4.0, 4.0]]> : tensor<2x2xf32>
  %b = arith.constant dense<[1.0, 3.0]> : tensor<2xf32>
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<4x2xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
  %m = linalg.matmul ins(%in, %w : tensor<4x2xf32>, tensor<2x2xf32>)
       outs(%f : tensor<4x2xf32>) -> tensor<4x2xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>],
                       iterator_types = ["parallel","parallel"]}
    ins(%m, %b : tensor<4x2xf32>, tensor<2xf32>) outs(%e : tensor<4x2xf32>) {
  ^bb0(%x: f32, %k: f32, %o: f32):
    %p = arith.addf %x, %k : f32
    linalg.yield %p : f32
  } -> tensor<4x2xf32>
  return %r : tensor<4x2xf32>
}

// The layout rewrite absorbs a convolution's back-transpose into whatever
// elementwise operation follows it, so the batch norm that reaches this pass
// reads NHWC and writes NCHW. Removing it must give the transpose back rather
// than drop it -- and a frontend writes a constant 0 rather than the dimension
// where an axis has extent 1, which the permutation has to allow for.
// CHECK-LABEL: func.func @carries_a_relayout
// CHECK:         %[[C:.*]] = linalg.conv_2d_nhwc_hwcf
// CHECK:         linalg.transpose ins(%[[C]] : tensor<1x4x4x2xf32>)
// CHECK-SAME:      permutation = [0, 3, 1, 2]
// CHECK-NOT:     math.rsqrt
func.func @carries_a_relayout(%in: tensor<1x4x4x1xf32>) -> tensor<1x2x4x4xf32> {
  %w = arith.constant dense<[[[[2.0, 4.0]]]]> : tensor<1x1x1x2xf32>
  %gamma = arith.constant dense<[3.0, 2.0]> : tensor<2xf32>
  %beta = arith.constant dense<[1.0, -1.0]> : tensor<2xf32>
  %mean = arith.constant dense<[2.0, 0.5]> : tensor<2xf32>
  %var = arith.constant dense<[0.75, 3.75]> : tensor<2xf32>
  %eps = arith.constant 2.500000e-01 : f32
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x4x4x2xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : tensor<1x4x4x1xf32>, tensor<1x1x1x2xf32>)
    outs(%f : tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32>
  %o = tensor.empty() : tensor<1x2x4x4xf32>
  %bn = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(0,d2,d3,d1)>,
                                         affine_map<(d0,d1,d2,d3)->(d1)>,
                                         affine_map<(d0,d1,d2,d3)->(d1)>,
                                         affine_map<(d0,d1,d2,d3)->(d1)>,
                                         affine_map<(d0,d1,d2,d3)->(d1)>,
                                         affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c, %gamma, %beta, %mean, %var : tensor<1x4x4x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
    outs(%o : tensor<1x2x4x4xf32>) {
  ^bb0(%x: f32, %g: f32, %b: f32, %m: f32, %v: f32, %out: f32):
    %ve = arith.addf %v, %eps : f32
    %inv = math.rsqrt %ve : f32
    %d = arith.subf %x, %m : f32
    %s = arith.mulf %d, %inv : f32
    %sg = arith.mulf %s, %g : f32
    %r = arith.addf %sg, %b : f32
    linalg.yield %r : f32
  } -> tensor<1x2x4x4xf32>
  return %bn : tensor<1x2x4x4xf32>
}

// The same value for every channel is a fill, not a broadcast. It is what an
// average pool leaves: `--average-pool-to-contraction` makes it a depthwise
// convolution with an all-ones filter, and this folds the divide by the
// window's size into that filter with nothing to add afterwards. Written as a
// broadcast the zero would be hidden from --force-quantized-matmul, which then
// dequantizes and adds it back -- a stray `+ 0.0` in the middle of the
// requantization, which the requantization matcher does not walk, and the
// convolution stops folding.
// CHECK-LABEL: func.func @uniform_bias_is_a_fill
// CHECK:         %[[W:.*]] = arith.constant dense<2.500000e-01> : tensor<2x2x2xf32>
// CHECK:         %[[F:.*]] = linalg.fill ins(%{{.*}} : f32)
// CHECK:         linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      ins(%{{.*}}, %[[W]]
// CHECK-SAME:      outs(%[[F]]
// CHECK-NOT:     linalg.generic
func.func @uniform_bias_is_a_fill(%in: tensor<1x4x4x2xf32>) -> tensor<1x2x2x2xf32> {
  %ones = arith.constant dense<1.0> : tensor<2x2x2xf32>
  %four = arith.constant 4.000000e+00 : f32
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %c = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%in, %ones : tensor<1x4x4x2xf32>, tensor<2x2x2xf32>)
    outs(%f : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %avg = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                                          affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x2x2x2xf32>) outs(%e : tensor<1x2x2x2xf32>) {
  ^bb0(%x: f32, %o: f32):
    %d = arith.divf %x, %four : f32
    linalg.yield %d : f32
  } -> tensor<1x2x2x2xf32>
  return %avg : tensor<1x2x2x2xf32>
}

// A grouped convolution arrives as G convolutions joined by a concatenation,
// with one batch norm sitting on the join -- so the fold finds a concatenation
// above it rather than a contraction, and not one of the G convolutions ever
// loses its batch norm. Each output channel belongs to exactly one group, so
// the per-channel constants cut along the joined axis.
//
// The filters are slices of one constant too, and a slice is not a constant:
// they are materialized here so the fold can read the weights it scales.
//
// Group 0 takes channels 0..1 (a = 3 and 1, b = -5 and -1.5 as above), group 1
// channels 2..3 (gamma 4 over sqrt(4) = 2, b = 2 - 1*2 = 0; gamma 3 over
// sqrt(0.25) = 6, b = 0 - 3*6 = -18).
//
// CHECK-LABEL: func.func @grouped
//   Weights, scaled by their own group's channels:
// CHECK-DAG:   %[[W0:.*]] = arith.constant dense<{{\[}}{{\[}}{{\[}}[3.000000e+00, 1.000000e+00]
// CHECK-DAG:   %[[W1:.*]] = arith.constant dense<{{\[}}{{\[}}{{\[}}[2.000000e+00, 6.000000e+00]
//   Biases, one broadcast per group:
// CHECK-DAG:   %[[B0:.*]] = arith.constant dense<[-5.000000e+00, -1.500000e+00]>
// CHECK-DAG:   %[[B1:.*]] = arith.constant dense<[0.000000e+00, -1.800000e+01]>
// CHECK:       linalg.conv_2d_nhwc_hwcf {{.*}}ins(%{{.*}}, %[[W0]] 
// CHECK:       linalg.conv_2d_nhwc_hwcf {{.*}}ins(%{{.*}}, %[[W1]] 
//   The join is on its own axis and the relayout the batch norm was carrying is
//   left as one copy over it, not one per group.
// CHECK:       tensor.concat dim(3)
// CHECK:       linalg.generic
// CHECK-NEXT:  ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:    linalg.yield %[[IN]]
// CHECK-NOT:   arith.subf
#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#nchw = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#gchan = affine_map<(d0, d1, d2, d3) -> (d1)>
func.func @grouped(%x: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %eps = arith.constant 2.500000e-01 : f32
  %filters = arith.constant dense<1.0> : tensor<1x1x2x4xf32>
  %gamma = arith.constant dense<[3.0, 2.0, 4.0, 3.0]> : tensor<4xf32>
  %beta  = arith.constant dense<[1.0, -1.0, 2.0, 0.0]> : tensor<4xf32>
  %mean  = arith.constant dense<[2.0, 5.000000e-01, 1.0, 3.0]> : tensor<4xf32>
  %var   = arith.constant dense<[7.500000e-01, 3.750000e+00, 3.750000e+00, 0.0]> : tensor<4xf32>

  %e = tensor.empty() : tensor<1x4x4x2xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32>
  %x0 = tensor.extract_slice %x[0, 0, 0, 0] [1, 4, 4, 2] [1, 1, 1, 1] : tensor<1x4x4x4xf32> to tensor<1x4x4x2xf32>
  %x1 = tensor.extract_slice %x[0, 0, 0, 2] [1, 4, 4, 2] [1, 1, 1, 1] : tensor<1x4x4x4xf32> to tensor<1x4x4x2xf32>
  %w0 = tensor.extract_slice %filters[0, 0, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x1x2x4xf32> to tensor<1x1x2x2xf32>
  %w1 = tensor.extract_slice %filters[0, 0, 0, 2] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x1x2x4xf32> to tensor<1x1x2x2xf32>
  %c0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%x0, %w0 : tensor<1x4x4x2xf32>, tensor<1x1x2x2xf32>) outs(%f : tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32>
  %c1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%x1, %w1 : tensor<1x4x4x2xf32>, tensor<1x1x2x2xf32>) outs(%f : tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32>
  %j = tensor.concat dim(3) %c0, %c1 : (tensor<1x4x4x2xf32>, tensor<1x4x4x2xf32>) -> tensor<1x4x4x4xf32>

  %o = tensor.empty() : tensor<1x4x4x4xf32>
  %bn = linalg.generic {indexing_maps = [#nhwc, #gchan, #gchan, #gchan, #gchan, #nchw],
                        iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%j, %gamma, %beta, %mean, %var : tensor<1x4x4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
    outs(%o : tensor<1x4x4x4xf32>) {
  ^bb0(%in: f32, %g: f32, %b: f32, %m: f32, %v: f32, %out: f32):
    %1 = arith.addf %v, %eps : f32
    %2 = math.rsqrt %1 : f32
    %3 = arith.subf %in, %m : f32
    %4 = arith.mulf %3, %2 : f32
    %5 = arith.mulf %4, %g : f32
    %6 = arith.addf %5, %b : f32
    linalg.yield %6 : f32
  } -> tensor<1x4x4x4xf32>
  return %bn : tensor<1x4x4x4xf32>
}
