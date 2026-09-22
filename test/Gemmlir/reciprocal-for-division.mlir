// RUN: gemmlir-opt --reciprocal-for-division %s | FileCheck %s

// A quantization divides by its scale. On this in-order core the divide is
// most of the loop; the reciprocal is a compile-time constant.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @quantize
// CHECK:         %[[R:.*]] = arith.constant 1.250000e+01 : f32
// CHECK:         linalg.generic
// CHECK:           arith.mulf %{{.*}}, %[[R]] : f32
// CHECK-NOT:     arith.divf
func.func @quantize(%in: tensor<4x8xf32>, %out: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %scale = arith.constant 0.08 : f32
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%in: tensor<4x8xf32>) outs(%out: tensor<4x8xf32>) {
  ^bb0(%a: f32, %b: f32):
    %d = arith.divf %a, %scale : f32
    linalg.yield %d : f32
  } -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// The reciprocal is folded at the divide's own width, not in double and
// truncated.

// CHECK-LABEL: func @half
// CHECK:         %[[H:.*]] = arith.constant 2.500000e-01 : f16
// CHECK:         arith.mulf %{{.*}}, %[[H]] : f16
func.func @half(%a: f16) -> f16 {
  %c = arith.constant 4.0 : f16
  %0 = arith.divf %a, %c : f16
  return %0 : f16
}

// A divisor that is not a constant stays a divide: there is nothing to fold.

// CHECK-LABEL: func @variable
// CHECK:         arith.divf
func.func @variable(%a: f32, %b: f32) -> f32 {
  %0 = arith.divf %a, %b : f32
  return %0 : f32
}

// Neither does a scale of zero or an infinity -- the reciprocal is not finite,
// and the multiply would not agree with the divide on ordinary values.

// CHECK-LABEL: func @zero
// CHECK:         arith.divf
func.func @zero(%a: f32) -> f32 {
  %c = arith.constant 0.0 : f32
  %0 = arith.divf %a, %c : f32
  return %0 : f32
}

// CHECK-LABEL: func @huge
// CHECK:         arith.divf
func.func @huge(%a: f32) -> f32 {
  %c = arith.constant 0x7F7FFFFF : f32
  %0 = arith.divf %a, %c : f32
  return %0 : f32
}
