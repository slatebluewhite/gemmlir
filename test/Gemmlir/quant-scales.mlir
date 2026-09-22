// Symmetric int8 scales are max|x|/127. A constant operand is measured here; an
// activation cannot be, so a calibration step annotates the operation.

// RUN: gemmlir-opt --force-quantized-matmul %s | FileCheck %s
// RUN: gemmlir-opt --force-quantized-matmul="activation-scale=0.05" %s \
// RUN: | FileCheck %s --check-prefix=OPT

#t = affine_map<(d0, d1) -> (d1, d0)>
#i = affine_map<(d0, d1) -> (d0, d1)>

// The weight is a constant reached through a transpose, whose largest magnitude
// is 0.254: 0.254/127 = 0.002. The activation is not constant, so the
// annotation on the operation decides its scale.
// Because the weight's scale is known at compile time, its quantization is
// folded (see fold-weight-quant.mlir) and the chosen scale shows up in the
// folded bytes -- 0.254/0.002 = 127, 0.1/0.002 = 50 -- rather than in a type.
// CHECK-LABEL: func.func @annotated
// CHECK-DAG:     arith.constant dense<{{\[}}[127, 50, 0, 25], [-50, -127, 50, 0], [0, 25, 100, -100], [100, 0, -50, 50]]> : tensor<4x4xi8>
// CHECK-DAG:     !quant.uniform<i8:f32, 1.000000e-02>
// The accumulator carries the product of the two.
// CHECK-DAG:     !quant.uniform<i32:f32, 2.0000000{{[0-9]*}}E-5>
func.func @annotated(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %w = arith.constant dense<[[0.254, -0.1, 0.0, 0.2],
                             [0.1, -0.254, 0.05, 0.0],
                             [0.0, 0.1, 0.2, -0.1],
                             [0.05, 0.0, -0.2, 0.1]]> : tensor<4x4xf32>
  %e = tensor.empty() : tensor<4x4xf32>
  %wt = linalg.generic {indexing_maps = [#t, #i], iterator_types = ["parallel", "parallel"]}
    ins(%w : tensor<4x4xf32>) outs(%e : tensor<4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x4xf32>
  %z = arith.constant 0.0 : f32
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
  %r = linalg.matmul {gemmlir.activation_scale = 1.000000e-02 : f64}
       ins(%a, %wt : tensor<4x4xf32>, tensor<4x4xf32>) outs(%f : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// Without an annotation the pass option stands in, but the weight is still
// measured rather than assumed.
// OPT-LABEL: func.func @unannotated
// OPT-DAG:     arith.constant dense<{{\[}}[127, -50, 0, 100], [50, -127, 25, 0], [0, 50, 100, -50], [25, 0, -100, 50]]> : tensor<4x4xi8>
// OPT-DAG:     !quant.uniform<i8:f32, 5.000000e-02>
func.func @unannotated(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %w = arith.constant dense<[[0.254, -0.1, 0.0, 0.2],
                             [0.1, -0.254, 0.05, 0.0],
                             [0.0, 0.1, 0.2, -0.1],
                             [0.05, 0.0, -0.2, 0.1]]> : tensor<4x4xf32>
  %e = tensor.empty() : tensor<4x4xf32>
  %z = arith.constant 0.0 : f32
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
  %r = linalg.matmul ins(%a, %w : tensor<4x4xf32>, tensor<4x4xf32>) outs(%f : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
