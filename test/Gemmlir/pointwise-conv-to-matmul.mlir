// `tiled_conv_auto` writes `elem_t`: the accelerator's convolution always
// requantizes its accumulator to i8, and `gemmlir.conv2d_i8` is the only shape
// there is. A convolution whose result the model *returns* has no
// requantization to fold -- the value leaves in f32 -- so it has nowhere to go
// and stays a scalar loop. Every detector and every model with an auxiliary
// head ends in exactly that.
//
// A 1x1 convolution over NHWC is a matmul: the pixels are the rows and the
// channels are the contraction. `tiled_matmul_auto` has a form that writes the
// i32 accumulator, so as a matmul it offloads.

// RUN: gemmlir-opt --pointwise-conv-to-matmul --canonicalize %s | FileCheck %s
// RUN: gemmlir-opt --pointwise-conv-to-matmul --canonicalize \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
// RUN:   --buffer-deallocation-pipeline --convert-linalg-to-gemmlir %s \
// RUN: | FileCheck %s --check-prefix=GEMMINI

// A detection head: the 1x1 convolution's result is dequantized and returned.
// CHECK-LABEL: func @head
// CHECK:         %[[R:.*]] = tensor.collapse_shape %arg0 {{\[}}[0, 1, 2], [3]{{\]}}
// CHECK-SAME:      tensor<1x4x4x16xi8> into tensor<16x16xi8>
// CHECK:         %[[W:.*]] = tensor.collapse_shape %arg1 {{\[}}[0, 1, 2], [3]{{\]}}
// CHECK-SAME:      tensor<1x1x16x8xi8> into tensor<16x8xi8>
// CHECK:         %[[M:.*]] = linalg.matmul ins(%[[R]], %[[W]] : tensor<16x16xi8>, tensor<16x8xi8>)
// CHECK:         tensor.expand_shape %[[M]]
// CHECK-SAME:      tensor<16x8xi32> into tensor<1x4x4x8xi32>
// CHECK-NOT:     linalg.conv_2d_nhwc_hwcf

// GEMMINI-LABEL: func @head
// GEMMINI:         gemmlir.matmul_i8
// GEMMINI-NOT:     linalg.conv_2d_nhwc_hwcf
func.func @head(%in: tensor<1x4x4x16xi8>, %f: tensor<1x1x16x8xi8>) -> tensor<1x4x4x8xf32> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x4x4x8xi32>
  %acc = linalg.fill ins(%z : i32) outs(%e : tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%in, %f : tensor<1x4x4x16xi8>, tensor<1x1x16x8xi8>)
       outs(%acc : tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %o = tensor.empty() : tensor<1x4x4x8xf32>
  %d = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%c : tensor<1x4x4x8xi32>) outs(%o : tensor<1x4x4x8xf32>) {
  ^bb0(%a: i32, %b: f32):
    %s = arith.constant 1.250000e-02 : f32
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x4x4x8xf32>
  return %d : tensor<1x4x4x8xf32>
}

// -----

// A 1x1 convolution followed by a requantization is left alone: `conv2d_i8`
// takes the bias, the activation and a fused pooling with it, and on this board
// the convolution call is the faster of the two.
// CHECK-LABEL: func @requantizes
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:     linalg.matmul

// GEMMINI-LABEL: func @requantizes
func.func @requantizes(%in: tensor<1x4x4x16xi8>, %f: tensor<1x1x16x8xi8>) -> tensor<1x4x4x8xi8> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x4x4x8xi32>
  %acc = linalg.fill ins(%z : i32) outs(%e : tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%in, %f : tensor<1x4x4x16xi8>, tensor<1x1x16x8xi8>)
       outs(%acc : tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %o = tensor.empty() : tensor<1x4x4x8xi8>
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%c : tensor<1x4x4x8xi32>) outs(%o : tensor<1x4x4x8xi8>) {
  ^bb0(%a: i32, %b: i8):
    %s = arith.constant 1.250000e-02 : f32
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    %r = arith.fptosi %m : f32 to i8
    linalg.yield %r : i8
  } -> tensor<1x4x4x8xi8>
  return %q : tensor<1x4x4x8xi8>
}

// -----

// A 3x3 convolution is not a matmul: each output pixel reads nine input pixels.
// CHECK-LABEL: func @not_pointwise
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:     linalg.matmul
func.func @not_pointwise(%in: tensor<1x6x6x16xi8>, %f: tensor<3x3x16x8xi8>) -> tensor<1x4x4x8xf32> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x4x4x8xi32>
  %acc = linalg.fill ins(%z : i32) outs(%e : tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%in, %f : tensor<1x6x6x16xi8>, tensor<3x3x16x8xi8>)
       outs(%acc : tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %o = tensor.empty() : tensor<1x4x4x8xf32>
  %d = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%c : tensor<1x4x4x8xi32>) outs(%o : tensor<1x4x4x8xf32>) {
  ^bb0(%a: i32, %b: f32):
    %s = arith.constant 1.250000e-02 : f32
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x4x4x8xf32>
  return %d : tensor<1x4x4x8xf32>
}

// -----

// A strided 1x1 convolution subsamples: the output pixels are not the input
// pixels, so collapsing both to one row axis would be wrong.
// CHECK-LABEL: func @strided
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:     linalg.matmul
func.func @strided(%in: tensor<1x8x8x16xi8>, %f: tensor<1x1x16x8xi8>) -> tensor<1x4x4x8xf32> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x4x4x8xi32>
  %acc = linalg.fill ins(%z : i32) outs(%e : tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
       ins(%in, %f : tensor<1x8x8x16xi8>, tensor<1x1x16x8xi8>)
       outs(%acc : tensor<1x4x4x8xi32>) -> tensor<1x4x4x8xi32>
  %o = tensor.empty() : tensor<1x4x4x8xf32>
  %d = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%c : tensor<1x4x4x8xi32>) outs(%o : tensor<1x4x4x8xf32>) {
  ^bb0(%a: i32, %b: f32):
    %s = arith.constant 1.250000e-02 : f32
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x4x4x8xf32>
  return %d : tensor<1x4x4x8xf32>
}
