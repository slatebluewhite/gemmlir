// Taking the fallback scale is said out loud. One layer quantized at a number
// nobody measured used to be invisible -- `cnn_i2c` reached the board with its
// first convolution in f32, and `gapb`'s global average pool quantized at
// 0.02, and the only way to notice either was to count loops in the final IR.

// RUN: gemmlir-opt --force-quantized-matmul %s 2>&1 | FileCheck %s

// CHECK: warning: no gemmlir.activation_scale; quantizing this contraction at the fallback
// CHECK-SAME: --raise-contraction-to-matmul before calibrating
func.func @unmeasured(%a: tensor<8x16xf32>, %b: tensor<16x4xf32>,
                      %init: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = linalg.matmul ins(%a, %b : tensor<8x16xf32>, tensor<16x4xf32>)
                     outs(%init : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// A calibrated one says nothing.
// CHECK-NOT: warning
func.func @measured(%a: tensor<8x16xf32>, %b: tensor<16x4xf32>,
                    %init: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = linalg.matmul {gemmlir.activation_scale = 1.250000e-02 : f64}
       ins(%a, %b : tensor<8x16xf32>, tensor<16x4xf32>)
       outs(%init : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}
