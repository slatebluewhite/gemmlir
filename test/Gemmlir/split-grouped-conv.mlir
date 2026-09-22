// `linalg.conv_2d_ngchw_gfchw` is what torch-mlir emits for
// `nn.Conv2d(..., groups=G)` -- the grouped convolution of ResNeXt, RegNet and
// every "cardinality" block. Gemmini has no notion of groups and does not need
// one: group `g` reads only input channels [g*C/G, (g+1)*C/G) and writes only
// output channels [g*F/G, (g+1)*F/G), so it is G convolutions that never see
// each other's data.

// RUN: gemmlir-opt --split-grouped-conv --canonicalize %s | FileCheck %s
// RUN: gemmlir-opt --split-grouped-conv --canonicalize --conv-nchw-to-nhwc \
// RUN:   --canonicalize %s | FileCheck %s --check-prefix=LAYOUT

// Two groups of four input and three output channels each.
//
// The slices are taken in NHWC and the joins are a concatenation. Both matter:
// relayouting after the split would give every group its own pair of
// transposes, and joining with `tensor.insert_slice` would leave every group's
// tail in f32 where the concatenation's is pushed back into each branch.
// CHECK-LABEL: func @grouped
// CHECK:         %[[IN:.*]] = tensor.collapse_shape %arg0 {{\[}}[0], [1, 2], [3], [4]{{\]}}
// CHECK-SAME:      tensor<1x2x4x6x6xf32> into tensor<1x8x6x6xf32>
// CHECK:         %[[NHWC:.*]] = linalg.transpose ins(%[[IN]]
// CHECK-SAME:      permutation = [0, 2, 3, 1]
// CHECK:         %[[F:.*]] = tensor.collapse_shape %arg1 {{\[}}[0, 1], [2], [3], [4]{{\]}}
// CHECK-SAME:      tensor<2x3x4x3x3xf32> into tensor<6x4x3x3xf32>
// CHECK:         %[[HWCF:.*]] = linalg.transpose ins(%[[F]]
// CHECK-SAME:      permutation = [2, 3, 1, 0]
// CHECK:         tensor.extract_slice %[[NHWC]][0, 0, 0, 0] [1, 6, 6, 4]
// CHECK:         tensor.extract_slice %[[HWCF]][0, 0, 0, 0] [3, 3, 4, 3]
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK:         tensor.extract_slice %[[NHWC]][0, 0, 0, 4] [1, 6, 6, 4]
// CHECK:         tensor.extract_slice %[[HWCF]][0, 0, 0, 3] [3, 3, 4, 3]
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK:         tensor.concat dim(3)
// CHECK-NOT:     linalg.conv_2d_ngchw_gfchw

// One transpose in and one out, whatever G is -- the filter's folds away
// because it is a constant, and no group has one of its own.
// LAYOUT-LABEL: func @grouped
// LAYOUT-COUNT-3: linalg.transpose
// LAYOUT-NOT:   linalg.transpose
func.func @grouped(%in: tensor<1x2x4x6x6xf32>, %f: tensor<2x3x4x3x3xf32>)
    -> tensor<1x6x4x4xf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x2x3x4x4xf32>
  %acc = linalg.fill ins(%z : f32) outs(%e : tensor<1x2x3x4x4xf32>) -> tensor<1x2x3x4x4xf32>
  %c = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%in, %f : tensor<1x2x4x6x6xf32>, tensor<2x3x4x3x3xf32>)
       outs(%acc : tensor<1x2x3x4x4xf32>) -> tensor<1x2x3x4x4xf32>
  // torch-mlir always collapses the groups straight back out, and the pass
  // replaces this rather than expanding into it: an expand and a collapse that
  // cancel would still sit between this convolution's transpose back to NCHW
  // and the next one's transpose to NHWC, and stop those two cancelling.
  %r = tensor.collapse_shape %c [[0], [1, 2], [3], [4]] : tensor<1x2x3x4x4xf32> into tensor<1x6x4x4xf32>
  return %r : tensor<1x6x4x4xf32>
}

// -----

// One group is an ordinary convolution wearing a 5-D shape; the 4-D form
// already has a path, so leave it to it.
// CHECK-LABEL: func @single_group
// CHECK:         linalg.conv_2d_ngchw_gfchw
// CHECK-NOT:     tensor.concat
func.func @single_group(%in: tensor<1x1x4x6x6xf32>, %f: tensor<1x3x4x3x3xf32>)
    -> tensor<1x1x3x4x4xf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x1x3x4x4xf32>
  %acc = linalg.fill ins(%z : f32) outs(%e : tensor<1x1x3x4x4xf32>) -> tensor<1x1x3x4x4xf32>
  %c = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%in, %f : tensor<1x1x4x6x6xf32>, tensor<1x3x4x3x3xf32>)
       outs(%acc : tensor<1x1x3x4x4xf32>) -> tensor<1x1x3x4x4xf32>
  return %c : tensor<1x1x3x4x4xf32>
}

// -----

// A dynamic shape is not something this can slice; torch-mlir's own output
// arrives that way and `normalize()` in scripts/calibrate.py puts the static
// shapes back before anything here sees it.
// CHECK-LABEL: func @dynamic
// CHECK:         linalg.conv_2d_ngchw_gfchw
func.func @dynamic(%in: tensor<?x2x4x6x6xf32>, %f: tensor<2x3x4x3x3xf32>,
                   %acc: tensor<?x2x3x4x4xf32>) -> tensor<?x2x3x4x4xf32> {
  %c = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%in, %f : tensor<?x2x4x6x6xf32>, tensor<2x3x4x3x3xf32>)
       outs(%acc : tensor<?x2x3x4x4xf32>) -> tensor<?x2x3x4x4xf32>
  return %c : tensor<?x2x3x4x4xf32>
}
