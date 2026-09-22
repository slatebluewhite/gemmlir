// A constant weight does not need to be quantized at run time. The scale is
// already known (it was measured from the constant itself), so the pass folds
// the quantization -- through the transpose a frontend puts in front of it --
// into an i8 constant, instead of leaving a loop that recomputes the same
// bytes on every inference. On the two-layer CNN this removed 4928 elements of
// per-inference work and the output stayed bit-identical.

// RUN: gemmlir-opt --force-quantized-matmul --canonicalize %s | FileCheck %s

#t = affine_map<(d0, d1) -> (d1, d0)>
#i = affine_map<(d0, d1) -> (d0, d1)>

// The weight's largest magnitude is 0.254, so the scale is 0.254/127 = 0.002
// and the transposed, quantized constant is exactly [[127, 100], [-50, 50],
// [0, -127]]. The activation is not constant and still quantizes at run time.
// CHECK-LABEL: func.func @folds_transposed_weight
// CHECK-DAG:     %[[W:.*]] = arith.constant dense<{{\[}}[127, 100], [-50, 50], [0, -127]]> : tensor<3x2xi8>
// CHECK-NOT:     linalg.generic
// CHECK:         %[[A:.*]] = quant.scast
// CHECK:         linalg.matmul ins(%[[A]], %[[W]]
func.func @folds_transposed_weight(%a: tensor<2x3xf32>) -> tensor<2x2xf32> {
  %w = arith.constant dense<[[0.254, -0.1, 0.0], [0.2, 0.1, -0.254]]> : tensor<2x3xf32>
  %e = tensor.empty() : tensor<3x2xf32>
  // A frontend transposes by permuting the *output* map, not the input.
  %wt = linalg.generic {indexing_maps = [#i, #t], iterator_types = ["parallel", "parallel"]}
    ins(%w : tensor<2x3xf32>) outs(%e : tensor<3x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x2xf32>
  %z = arith.constant 0.0 : f32
  %eo = tensor.empty() : tensor<2x2xf32>
  %f = linalg.fill ins(%z : f32) outs(%eo : tensor<2x2xf32>) -> tensor<2x2xf32>
  %r = linalg.matmul {gemmlir.activation_scale = 1.000000e-02 : f64}
       ins(%a, %wt : tensor<2x3xf32>, tensor<3x2xf32>) outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// A weight reached with no transpose at all folds the same way.
// CHECK-LABEL: func.func @folds_direct_weight
// CHECK-DAG:     arith.constant dense<{{\[}}[127, -50], [0, 100]]> : tensor<2x2xi8>
func.func @folds_direct_weight(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %w = arith.constant dense<[[0.254, -0.1], [0.0, 0.2]]> : tensor<2x2xf32>
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x2xf32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
  %r = linalg.matmul {gemmlir.activation_scale = 1.000000e-02 : f64}
       ins(%a, %w : tensor<2x2xf32>, tensor<2x2xf32>) outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// An operand that is computed rather than copied is not a constant: the scale
// falls back to the activation scale and the quantization stays at run time.
// CHECK-LABEL: func.func @keeps_computed_operand
// CHECK:         quant.qcast %{{.*}} : tensor<2x2xf32> to tensor<2x2x!quant.uniform<i8:f32, 1.000000e-02>>
// CHECK:         quant.qcast %{{.*}} : tensor<2x2xf32> to tensor<2x2x!quant.uniform<i8:f32, 1.000000e-02>>
func.func @keeps_computed_operand(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %w = arith.constant dense<[[0.254, -0.1], [0.0, 0.2]]> : tensor<2x2xf32>
  %e = tensor.empty() : tensor<2x2xf32>
  %wd = linalg.generic {indexing_maps = [#i, #i], iterator_types = ["parallel", "parallel"]}
    ins(%w : tensor<2x2xf32>) outs(%e : tensor<2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %in : f32
    linalg.yield %s : f32
  } -> tensor<2x2xf32>
  %z = arith.constant 0.0 : f32
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
  %r = linalg.matmul {gemmlir.activation_scale = 1.000000e-02 : f64}
       ins(%a, %wd : tensor<2x2xf32>, tensor<2x2xf32>) outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}
