// A frontend lowers adaptive_avg_pool2d(x, 1) -- the head of every ResNet and
// MobileNet -- to a sum over the whole image followed by a divide. Summing a
// whole image per channel is a contraction: with the image read as (H*W, C) it
// is ones(1, H*W) x image, and that goes down the path that already exists.

// RUN: gemmlir-opt --average-pool-to-contraction --canonicalize %s | FileCheck %s

// NHWC already has the channel innermost, so the image becomes a matrix without
// moving anything. The divide by the pixel count is left where it is: it ends up
// in the accelerator call's own requantization scale.
// CHECK-LABEL: func.func @global
// CHECK-DAG:     %[[ONES:.*]] = arith.constant dense<1.000000e+00> : tensor<1x256xf32>
// CHECK:         %[[IMG:.*]] = tensor.collapse_shape %arg0 {{\[}}[0, 1, 2], [3]{{\]}}
// CHECK-SAME:      tensor<1x16x16x8xf32> into tensor<256x8xf32>
// CHECK:         %[[MM:.*]] = linalg.matmul ins(%[[ONES]], %[[IMG]]
// CHECK:         tensor.expand_shape %[[MM]]
// CHECK-NOT:     linalg.pooling
func.func @global(%in: tensor<1x16x16x8xf32>) -> tensor<1x1x1x8xf32> {
  %zero = arith.constant 0.0 : f32
  %win = tensor.empty() : tensor<16x16xf32>
  %e = tensor.empty() : tensor<1x1x1x8xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
  %p = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %win : tensor<1x16x16x8xf32>, tensor<16x16xf32>)
    outs(%f : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
  return %p : tensor<1x1x1x8xf32>
}

// A window that is not the whole image leaves several output pixels, and the
// collapse would mix them. It is a depthwise convolution whose filter is all
// ones -- which is what `tiled_conv_dw_auto` runs -- and the stride comes
// across with it.
// CHECK-LABEL: func.func @not_global
// CHECK:         %[[ONES:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2x8xf32>
// CHECK:         linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      strides = dense<2>
// CHECK-SAME:      ins(%arg0, %[[ONES]]
// CHECK-NOT:     linalg.pooling
func.func @not_global(%in: tensor<1x16x16x8xf32>) -> tensor<1x8x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %win = tensor.empty() : tensor<2x2xf32>
  %e = tensor.empty() : tensor<1x8x8x8xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  %p = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%in, %win : tensor<1x16x16x8xf32>, tensor<2x2xf32>)
    outs(%f : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %p : tensor<1x8x8x8xf32>
}

// More than one image in the buffer keeps its batch dimension: collapsing it
// into the contraction would sum across images.
// CHECK-LABEL: func.func @batched
// CHECK-DAG:     %[[ONES:.*]] = arith.constant dense<1.000000e+00> : tensor<4x1x256xf32>
// CHECK:         %[[IMG:.*]] = tensor.collapse_shape %arg0 {{\[}}[0], [1, 2], [3]{{\]}}
// CHECK-SAME:      tensor<4x16x16x8xf32> into tensor<4x256x8xf32>
// CHECK:         %[[MM:.*]] = linalg.batch_matmul ins(%[[ONES]], %[[IMG]]
// CHECK:         tensor.expand_shape %[[MM]]
// CHECK-NOT:     linalg.pooling
func.func @batched(%in: tensor<4x16x16x8xf32>) -> tensor<4x1x1x8xf32> {
  %zero = arith.constant 0.0 : f32
  %win = tensor.empty() : tensor<16x16xf32>
  %e = tensor.empty() : tensor<4x1x1x8xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<4x1x1x8xf32>) -> tensor<4x1x1x8xf32>
  %p = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %win : tensor<4x16x16x8xf32>, tensor<16x16xf32>)
    outs(%f : tensor<4x1x1x8xf32>) -> tensor<4x1x1x8xf32>
  return %p : tensor<4x1x1x8xf32>
}

// A sum that starts from something other than zero is not a sum of the image,
// and a matmul against ones would drop what was there.
// CHECK-LABEL: func.func @nonzero_init
// CHECK:         linalg.pooling_nhwc_sum
// CHECK-NOT:     linalg.matmul
// CHECK-NOT:     linalg.depthwise_conv_2d_nhwc_hwc
func.func @nonzero_init(%in: tensor<1x16x16x8xf32>, %init: tensor<1x1x1x8xf32>)
    -> tensor<1x1x1x8xf32> {
  %win = tensor.empty() : tensor<16x16xf32>
  %p = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %win : tensor<1x16x16x8xf32>, tensor<16x16xf32>)
    outs(%init : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
  return %p : tensor<1x1x1x8xf32>
}


// The contraction a windowed pool becomes is a new operation that no
// calibration annotated, and an unannotated operation falls back to the pass
// option's fixed scale -- the same fixed-scale problem calibration exists to
// solve. An average never leaves the range of what it averages, so the input's
// scale is a bound, and the first annotation above the pool is what to use: a
// relu or a bias usually sits in between and none of those widens the range.
// Measured on the board, `apb` went from 0.0140 relative L2 to 0.0056.
// CHECK-LABEL: func.func @carries_the_scale
// CHECK:         linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      gemmlir.activation_scale = 2.500000e-01
#idp = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @carries_the_scale(%in: tensor<1x18x18x8xf32>, %f: tensor<3x3x8x8xf32>,
                             %init: tensor<1x16x16x8xf32>) -> tensor<1x8x8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>,
                                 gemmlir.activation_scale = 2.500000e-01 : f64}
    ins(%in, %f : tensor<1x18x18x8xf32>, tensor<3x3x8x8xf32>)
    outs(%init : tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>
  %e = tensor.empty() : tensor<1x16x16x8xf32>
  %relu = linalg.generic {indexing_maps = [#idp, #idp], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x16x16x8xf32>) outs(%e : tensor<1x16x16x8xf32>) {
  ^bb0(%v: f32, %o: f32):
    %m = arith.maximumf %v, %zero : f32
    linalg.yield %m : f32
  } -> tensor<1x16x16x8xf32>
  %win = tensor.empty() : tensor<2x2xf32>
  %e1 = tensor.empty() : tensor<1x8x8x8xf32>
  %fl = linalg.fill ins(%zero : f32) outs(%e1 : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  %p = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%relu, %win : tensor<1x16x16x8xf32>, tensor<2x2xf32>)
    outs(%fl : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %p : tensor<1x8x8x8xf32>
}

// A global mean over hundreds of pixels collapses the range by roughly that
// factor, so the input's scale would leave most of the output's int8 range
// unused. That one is left to the calibration.
// CHECK-LABEL: func.func @global_does_not_carry_it
// CHECK:         linalg.matmul
// CHECK-NOT:     gemmlir.activation_scale
func.func @global_does_not_carry_it(%in: tensor<1x18x18x8xf32>, %f: tensor<3x3x8x8xf32>,
                                    %init: tensor<1x16x16x8xf32>) -> tensor<1x1x1x8xf32> {
  %zero = arith.constant 0.0 : f32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>,
                                 gemmlir.activation_scale = 2.500000e-01 : f64}
    ins(%in, %f : tensor<1x18x18x8xf32>, tensor<3x3x8x8xf32>)
    outs(%init : tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>
  %win = tensor.empty() : tensor<16x16xf32>
  %e1 = tensor.empty() : tensor<1x1x1x8xf32>
  %fl = linalg.fill ins(%zero : f32) outs(%e1 : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
  %p = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%c, %win : tensor<1x16x16x8xf32>, tensor<16x16xf32>)
    outs(%fl : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
  return %p : tensor<1x1x1x8xf32>
}
