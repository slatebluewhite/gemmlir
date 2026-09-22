// The whole --quantize pipeline honouring a calibration annotation, which is
// what scripts/calibrate.py writes. The script itself needs torch and cannot be
// tested here; this covers the contract it depends on.

// RUN: gemmlir-opt --force-quantized-matmul --canonicalize \
// RUN:   --lower-quant-ops --round-quantized-casts --strip-func-quant-types --canonicalize \
// RUN:   --convert-elementwise-to-linalg --canonicalize \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
// RUN:   --buffer-deallocation-pipeline --convert-linalg-to-gemmlir %s \
// RUN: | FileCheck %s

// The weight is quantized at compile time -- 0.254 / 0.002 = 127 -- so nothing
// converts it at run time.
// CHECK-DAG: memref.global {{.*}} : memref<4x4xi8> = dense<{{\[\[}}127, -50, 0, 100]
// The annotated scale is what the activation is divided by, and the
// accumulator's is the product of the two: 0.01 * 0.002 = 2e-5.
// CHECK-DAG: memref.global {{.*}} : memref<4x4xf32> = dense<0.00999999977>
// CHECK-DAG: memref.global {{.*}} : memref<4x4xf32> = dense<2.00000013E-5>

// CHECK-LABEL: func.func @annotated
// The activation's conversion rounds rather than truncates.
// CHECK:         math.roundeven
// CHECK:         gemmlir.matmul_i8
func.func @annotated(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %w = arith.constant dense<[[0.254, -0.1, 0.0, 0.2],
                             [0.1, -0.254, 0.05, 0.0],
                             [0.0, 0.1, 0.2, -0.1],
                             [0.05, 0.0, -0.2, 0.1]]> : tensor<4x4xf32>
  %e = tensor.empty() : tensor<4x4xf32>
  %z = arith.constant 0.0 : f32
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
  %r = linalg.matmul {gemmlir.activation_scale = 1.000000e-02 : f64}
       ins(%a, %w : tensor<4x4xf32>, tensor<4x4xf32>) outs(%f : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
