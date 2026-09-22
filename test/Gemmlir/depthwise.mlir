// A `groups == channels` convolution is what torch-mlir emits for the depthwise
// half of a MobileNet-style separable block, and the accelerator has its own
// entry point for it. The dialect has had `depthwise_conv2d_i8` and its
// lowering to `tiled_conv_dw_auto` all along; what was missing was any way to
// reach them from a frontend.

// RUN: gemmlir-opt --conv-nchw-to-nhwc --canonicalize %s | FileCheck %s --check-prefix=LAYOUT
// RUN: gemmlir-opt --convert-linalg-to-gemmlir %S/Inputs/depthwise-quantized.mlir | FileCheck %s --check-prefix=FOLD

// linalg counts a depthwise filter (C, KH, KW) in NCHW and (KH, KW, C) in NHWC.
// LAYOUT-LABEL: func.func @nchw_depthwise
// LAYOUT:         %[[IN:.*]] = linalg.transpose ins(%arg0
// LAYOUT-SAME:      permutation = [0, 2, 3, 1]
// LAYOUT:         linalg.depthwise_conv_2d_nhwc_hwc
// LAYOUT-SAME:      ins(%[[IN]], %{{.*}} : tensor<1x6x6x2xf32>, tensor<3x3x2xf32>)
// LAYOUT-NOT:     depthwise_conv_2d_nchw_chw
func.func @nchw_depthwise(%in: tensor<1x2x6x6xf32>, %f: tensor<2x3x3xf32>,
                          %init: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %c = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
     ins(%in, %f : tensor<1x2x6x6xf32>, tensor<2x3x3xf32>) outs(%init : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %c : tensor<1x2x4x4xf32>
}

// The runtime counts it (C, KH, KW) again, so the constant is permuted back at
// compile time rather than in a loop: filter[kh][kw][c] lands at
// [c][kh][kw], so 1,2 / 3,4 / 5,6 / 7,8 becomes 1,3,5,7 / 2,4,6,8.
// FOLD: memref.global "private" constant @[[G:.*]]_chw : memref<2x2x2xi8> = dense<{{\[}}{{\[}}[1, 3], [5, 7]], {{\[}}[2, 4], [6, 8]]]>
// FOLD-LABEL: func.func @quantized_depthwise
// FOLD-NOT:     linalg.depthwise_conv_2d_nhwc_hwc
// FOLD:         %[[W:.*]] = memref.get_global @[[G]]_chw
// FOLD:         gemmlir.depthwise_conv2d_i8(%arg0, %[[W]], %arg2)
// FOLD-SAME:      bias(%arg1 : memref<2xi32>)
// FOLD-SAME:      {act = #gemmlir.act<relu>, scale = 4.000000e-02 : f32}
