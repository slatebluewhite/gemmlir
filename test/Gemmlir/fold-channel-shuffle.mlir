// ShuffleNet's unit ends by interleaving the channels of the two halves it
// joined, which a frontend writes as `expand_shape` splitting the channel axis
// into (groups, rest), a transpose of those two, and `collapse_shape` putting
// them back. It is a fixed permutation of one axis, and a convolution reading a
// permuted input computes the same thing as the same convolution reading the
// original with its filter's input channels permuted the other way.

// RUN: gemmlir-opt --fold-channel-shuffle --canonicalize %s | FileCheck %s

// Two groups of three. The shuffle reads input channel `i*3 + j` into output
// channel `j*2 + i`, so it takes [0, 3, 1, 4, 2, 5] -- and the filter takes the
// **inverse**, [0, 2, 4, 1, 3, 5]. That is not the same list: the permutation
// is an involution only when the two group sizes are equal, so getting this the
// wrong way round is a wrong answer rather than a rearranged one, and it cost a
// relative L2 of 0.12 where 0.0036 was available.
//
// Reading it the other way: the result is sum_k W[k] * X[shuffle[k]], which
// collects to 10*X0 + 30*X1 + 50*X2 + 20*X3 + 40*X4 + 60*X5.
// CHECK-LABEL: func.func @channel_shuffle
// CHECK:         %[[W:.*]] = arith.constant dense<{{.*}}1.000000e+01{{.*}}3.000000e+01{{.*}}5.000000e+01{{.*}}2.000000e+01{{.*}}4.000000e+01{{.*}}6.000000e+01{{.*}}> : tensor<1x6x1x1xf32>
// CHECK:         linalg.conv_2d_nchw_fchw {{.*}}ins(%arg0, %[[W]]
// CHECK-NOT:     tensor.expand_shape
// CHECK-NOT:     tensor.collapse_shape
func.func @channel_shuffle(%x: tensor<1x6x4x4xf32>) -> tensor<1x1x4x4xf32> {
  %w = arith.constant dense<[[[[10.0]], [[20.0]], [[30.0]], [[40.0]], [[50.0]], [[60.0]]]]>
       : tensor<1x6x1x1xf32>
  %zero = arith.constant 0.0 : f32
  %e5 = tensor.empty() : tensor<1x3x2x4x4xf32>
  %expanded = tensor.expand_shape %x [[0], [1, 2], [3], [4]] output_shape [1, 2, 3, 4, 4]
              : tensor<1x6x4x4xf32> into tensor<1x2x3x4x4xf32>
  %swapped = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2,d3,d4)>,
                                              affine_map<(d0,d1,d2,d3,d4)->(d0,d2,d1,d3,d4)>],
                             iterator_types = ["parallel","parallel","parallel","parallel","parallel"]}
    ins(%expanded : tensor<1x2x3x4x4xf32>) outs(%e5 : tensor<1x3x2x4x4xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<1x3x2x4x4xf32>
  %collapsed = tensor.collapse_shape %swapped [[0], [1, 2], [3], [4]]
               : tensor<1x3x2x4x4xf32> into tensor<1x6x4x4xf32>
  %e = tensor.empty() : tensor<1x1x4x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  %c = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%collapsed, %w : tensor<1x6x4x4xf32>, tensor<1x6x1x1xf32>)
    outs(%f : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %c : tensor<1x1x4x4xf32>
}

// A convolution reads its input through the border it needs, and padding the
// spatial axes commutes with permuting the channels -- so the shuffle may be
// under one, and it is the pad's operand that gets rewired.
// CHECK-LABEL: func.func @shuffle_under_a_pad
// CHECK:         %[[P:.*]] = tensor.pad %arg0
// CHECK:         linalg.conv_2d_nchw_fchw {{.*}}ins(%[[P]]
// CHECK-NOT:     tensor.collapse_shape
func.func @shuffle_under_a_pad(%x: tensor<1x4x4x4xf32>) -> tensor<1x1x4x4xf32> {
  %w = arith.constant dense<1.0> : tensor<1x4x3x3xf32>
  %zero = arith.constant 0.0 : f32
  %e5 = tensor.empty() : tensor<1x2x2x4x4xf32>
  %expanded = tensor.expand_shape %x [[0], [1, 2], [3], [4]] output_shape [1, 2, 2, 4, 4]
              : tensor<1x4x4x4xf32> into tensor<1x2x2x4x4xf32>
  %swapped = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2,d3,d4)>,
                                              affine_map<(d0,d1,d2,d3,d4)->(d0,d2,d1,d3,d4)>],
                             iterator_types = ["parallel","parallel","parallel","parallel","parallel"]}
    ins(%expanded : tensor<1x2x2x4x4xf32>) outs(%e5 : tensor<1x2x2x4x4xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<1x2x2x4x4xf32>
  %collapsed = tensor.collapse_shape %swapped [[0], [1, 2], [3], [4]]
               : tensor<1x2x2x4x4xf32> into tensor<1x4x4x4xf32>
  %padded = tensor.pad %collapsed low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c2: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x4x4x4xf32> to tensor<1x4x6x6xf32>
  %e = tensor.empty() : tensor<1x1x4x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  %c = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%padded, %w : tensor<1x4x6x6xf32>, tensor<1x4x3x3xf32>)
    outs(%f : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %c : tensor<1x1x4x4xf32>
}

// A filter that is only known at run time cannot be permuted at compile time.
// CHECK-LABEL: func.func @runtime_filter
// CHECK:         tensor.collapse_shape
// CHECK:         linalg.conv_2d_nchw_fchw
func.func @runtime_filter(%x: tensor<1x6x4x4xf32>, %w: tensor<1x6x1x1xf32>)
    -> tensor<1x1x4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %e5 = tensor.empty() : tensor<1x3x2x4x4xf32>
  %expanded = tensor.expand_shape %x [[0], [1, 2], [3], [4]] output_shape [1, 2, 3, 4, 4]
              : tensor<1x6x4x4xf32> into tensor<1x2x3x4x4xf32>
  %swapped = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2,d3,d4)>,
                                              affine_map<(d0,d1,d2,d3,d4)->(d0,d2,d1,d3,d4)>],
                             iterator_types = ["parallel","parallel","parallel","parallel","parallel"]}
    ins(%expanded : tensor<1x2x3x4x4xf32>) outs(%e5 : tensor<1x3x2x4x4xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<1x3x2x4x4xf32>
  %collapsed = tensor.collapse_shape %swapped [[0], [1, 2], [3], [4]]
               : tensor<1x3x2x4x4xf32> into tensor<1x6x4x4xf32>
  %e = tensor.empty() : tensor<1x1x4x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  %c = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%collapsed, %w : tensor<1x6x4x4xf32>, tensor<1x6x1x1xf32>)
    outs(%f : tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %c : tensor<1x1x4x4xf32>
}
