// img2col leaves a contraction generic when the batch is 1, and everything
// downstream matches named operations. The unit axes are collapsed away and a
// linalg.matmul is left behind.

// RUN: gemmlir-opt --raise-contraction-to-matmul %s | FileCheck %s

#lhs = affine_map<(d0,d1,d2,d3) -> (d1, d3)>
#rhs = affine_map<(d0,d1,d2,d3) -> (d0, d3, d2)>
#out = affine_map<(d0,d1,d2,d3) -> (d0, d1, d2)>

// d0 is a batch of one, but because the left-hand side does not carry it linalg
// calls it a second `n` rather than a batch dimension -- which is why picking
// the dimension of each kind that actually has extent matters.
// CHECK-LABEL: func.func @unit_batch
// CHECK:         tensor.collapse_shape
// CHECK:         linalg.matmul
// CHECK:         tensor.expand_shape
// CHECK-NOT:     linalg.generic
func.func @unit_batch(%a: tensor<16x27xf32>, %b: tensor<1x27x196xf32>,
                      %o: tensor<1x16x196xf32>) -> tensor<1x16x196xf32> {
  %r = linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                       iterator_types = ["parallel","parallel","parallel","reduction"]}
    ins(%a, %b : tensor<16x27xf32>, tensor<1x27x196xf32>) outs(%o : tensor<1x16x196xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<1x16x196xf32>
  return %r : tensor<1x16x196xf32>
}

// A real batch is not a matmul the accelerator can take in one call, so it stays.
// CHECK-LABEL: func.func @real_batch
// CHECK-NOT:     linalg.matmul
// CHECK:         linalg.generic
func.func @real_batch(%a: tensor<16x27xf32>, %b: tensor<4x27x196xf32>,
                      %o: tensor<4x16x196xf32>) -> tensor<4x16x196xf32> {
  %r = linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                       iterator_types = ["parallel","parallel","parallel","reduction"]}
    ins(%a, %b : tensor<16x27xf32>, tensor<4x27x196xf32>) outs(%o : tensor<4x16x196xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<4x16x196xf32>
  return %r : tensor<4x16x196xf32>
}

// Whatever was written on the contraction belongs on the matmul that replaces
// it. `calibrate.py` leaves the layer's measured range there, and dropping it
// is silent -- the layer is simply never quantized and runs as an f32 loop
// nest. That is what `cnn_i2c`'s first convolution does: 16x196x27 multiply-adds
// in software, in a model whose other layers are all on the accelerator.

#w = affine_map<(b, m, n, k) -> (m, k)>
#x = affine_map<(b, m, n, k) -> (b, k, n)>
#y = affine_map<(b, m, n, k) -> (b, m, n)>

// CHECK-LABEL: func.func @carries_the_calibration
// CHECK:         linalg.matmul {gemmlir.activation_scale = 1.250000e-02 : f64}
func.func @carries_the_calibration(%a: tensor<16x27xf32>, %b: tensor<1x27x196xf32>,
                                   %o: tensor<1x16x196xf32>) -> tensor<1x16x196xf32> {
  %0 = linalg.generic {indexing_maps = [#w, #x, #y],
                       iterator_types = ["parallel", "parallel", "parallel", "reduction"],
                       gemmlir.activation_scale = 1.25e-02 : f64}
    ins(%a, %b : tensor<16x27xf32>, tensor<1x27x196xf32>)
    outs(%o : tensor<1x16x196xf32>) {
  ^bb0(%x0: f32, %y0: f32, %acc: f32):
    %m = arith.mulf %x0, %y0 : f32
    %s = arith.addf %m, %acc : f32
    linalg.yield %s : f32
  } -> tensor<1x16x196xf32>
  return %0 : tensor<1x16x196xf32>
}

// The right-hand range of an activation-times-activation contraction is its
// own attribute and travels the same way.

// CHECK-LABEL: func.func @carries_both_ranges
// CHECK:         linalg.matmul
// CHECK-SAME:      gemmlir.activation_scale = 2.000000e-02 : f64
// CHECK-SAME:      gemmlir.rhs_activation_scale = 3.000000e-02 : f64
func.func @carries_both_ranges(%a: tensor<16x27xf32>, %b: tensor<1x27x196xf32>,
                               %o: tensor<1x16x196xf32>) -> tensor<1x16x196xf32> {
  %0 = linalg.generic {indexing_maps = [#w, #x, #y],
                       iterator_types = ["parallel", "parallel", "parallel", "reduction"],
                       gemmlir.activation_scale = 2.0e-02 : f64,
                       gemmlir.rhs_activation_scale = 3.0e-02 : f64}
    ins(%a, %b : tensor<16x27xf32>, tensor<1x27x196xf32>)
    outs(%o : tensor<1x16x196xf32>) {
  ^bb0(%x0: f32, %y0: f32, %acc: f32):
    %m = arith.mulf %x0, %y0 : f32
    %s = arith.addf %m, %acc : f32
    linalg.yield %s : f32
  } -> tensor<1x16x196xf32>
  return %0 : tensor<1x16x196xf32>
}
