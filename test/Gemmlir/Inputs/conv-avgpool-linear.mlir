// A convolution, a 2x2 average pool, and a linear layer -- the shape of every
// small classifier, and of `apb` in the model set. The frontend pools in f32
// and requantizes afterwards, so without `--average-pool-to-contraction` the
// convolution's tail has no i8 result to fold into and both it and the pool
// stay scalar loops.
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>
func.func @classify(%in: tensor<1x10x10x4xf32>) -> tensor<1x4xf32> {
  %zero = arith.constant 0.0 : f32
  %quarter = arith.constant 2.500000e-01 : f32
  %filter = arith.constant dense<2.000000e-02> : tensor<3x3x4x4xf32>
  %bias = arith.constant dense<[1.000000e-01, -2.000000e-01, 5.000000e-02, 0.000000e+00]> : tensor<4xf32>
  %weights = arith.constant dense<3.000000e-02> : tensor<64x4xf32>
  %fcbias = arith.constant dense<[0.000000e+00, 1.000000e-01, -1.000000e-01, 2.000000e-01]> : tensor<4xf32>

  %e0 = tensor.empty() : tensor<1x8x8x4xf32>
  %b = linalg.generic {indexing_maps = [#chan, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%bias : tensor<4xf32>) outs(%e0 : tensor<1x8x8x4xf32>) {
  ^bb0(%v: f32, %o: f32):
    linalg.yield %v : f32
  } -> tensor<1x8x8x4xf32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>,
                                 gemmlir.activation_scale = 5.000000e-01 : f64}
    ins(%in, %filter : tensor<1x10x10x4xf32>, tensor<3x3x4x4xf32>)
    outs(%b : tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>

  %win = tensor.empty() : tensor<2x2xf32>
  %e1 = tensor.empty() : tensor<1x4x4x4xf32>
  %fl = linalg.fill ins(%zero : f32) outs(%e1 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %p = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%c, %win : tensor<1x8x8x4xf32>, tensor<2x2xf32>)
    outs(%fl : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %e2 = tensor.empty() : tensor<1x4x4x4xf32>
  %avg = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%p : tensor<1x4x4x4xf32>) outs(%e2 : tensor<1x4x4x4xf32>) {
  ^bb0(%v: f32, %o: f32):
    %m = arith.mulf %v, %quarter : f32
    linalg.yield %m : f32
  } -> tensor<1x4x4x4xf32>

  %flat = tensor.collapse_shape %avg [[0], [1, 2, 3]] : tensor<1x4x4x4xf32> into tensor<1x64xf32>
  %e3 = tensor.empty() : tensor<1x4xf32>
  %fb = linalg.generic {indexing_maps = [#col, #id2], iterator_types = ["parallel","parallel"]}
    ins(%fcbias : tensor<4xf32>) outs(%e3 : tensor<1x4xf32>) {
  ^bb0(%v: f32, %o: f32):
    linalg.yield %v : f32
  } -> tensor<1x4xf32>
  %fc = linalg.matmul {gemmlir.activation_scale = 2.000000e+00 : f64}
    ins(%flat, %weights : tensor<1x64xf32>, tensor<64x4xf32>)
    outs(%fb : tensor<1x4xf32>) -> tensor<1x4xf32>
  return %fc : tensor<1x4xf32>
}
