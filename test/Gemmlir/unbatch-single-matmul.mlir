// `--convert-linalg-to-gemmlir` gives a `linalg.batch_matmul` a loop over the
// batch and a `memref.subview` per slice. Everything that fuses into a matmul
// -- the requantization, the bias, the activation -- looks at what follows the
// matmul in its own block, and what follows this one is the end of the loop
// body. So a batched matmul keeps every one of its tails as a separate scalar
// pass over the tensor.
//
// A batch of one has nothing to loop over. torch-mlir gives all five of an
// attention head's contractions as `linalg.batch_matmul`, and at batch 1 that
// was five matmuls whose quantization stayed scalar.

// RUN: gemmlir-opt --unbatch-single-matmul --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @one_batch
// Each reshape goes where its value is defined, so their order follows the
// arguments' definitions rather than the matmul's operands.
// CHECK-DAG:     %[[A:.*]] = tensor.collapse_shape %arg0 {{\[}}[0, 1], [2]{{\]}} : tensor<1x16x32xf32> into tensor<16x32xf32>
// CHECK-DAG:     %[[B:.*]] = tensor.collapse_shape %arg1 {{\[}}[0, 1], [2]{{\]}} : tensor<1x32x8xf32> into tensor<32x8xf32>
// CHECK:         %[[M:.*]] = linalg.matmul
// CHECK-SAME:      {gemmlir.activation_scale = 2.000000e-02 : f64}
// CHECK-SAME:      ins(%[[A]], %[[B]]
// CHECK:         tensor.expand_shape %[[M]]
// CHECK-NOT:     linalg.batch_matmul
func.func @one_batch(%a: tensor<1x16x32xf32>, %b: tensor<1x32x8xf32>,
                     %init: tensor<1x16x8xf32>) -> tensor<1x16x8xf32> {
  %m = linalg.batch_matmul {gemmlir.activation_scale = 2.000000e-02 : f64}
       ins(%a, %b : tensor<1x16x32xf32>, tensor<1x32x8xf32>)
       outs(%init : tensor<1x16x8xf32>) -> tensor<1x16x8xf32>
  return %m : tensor<1x16x8xf32>
}

// -----

// A real batch keeps its loop: the tails would have to be sliced to follow it
// in, which is a different transformation.
// CHECK-LABEL: func @four_batches
// CHECK:         linalg.batch_matmul
// CHECK-NOT:     linalg.matmul
func.func @four_batches(%a: tensor<4x16x32xf32>, %b: tensor<4x32x8xf32>,
                        %init: tensor<4x16x8xf32>) -> tensor<4x16x8xf32> {
  %m = linalg.batch_matmul ins(%a, %b : tensor<4x16x32xf32>, tensor<4x32x8xf32>)
       outs(%init : tensor<4x16x8xf32>) -> tensor<4x16x8xf32>
  return %m : tensor<4x16x8xf32>
}

// -----

// The elementwise work around it goes down to two dimensions as well, or the
// reshapes sit between a dequantization and the next quantization and keep the
// two from ever meeting.
// CHECK-LABEL: func @elementwise_follows
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK-SAME:      ins(%{{.*}} : tensor<16x8xf32>)
func.func @elementwise_follows(%x: tensor<1x16x8xf32>) -> tensor<1x16x8xf32> {
  %c = arith.constant 2.000000e+00 : f32
  %e = tensor.empty() : tensor<1x16x8xf32>
  %m = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%x : tensor<1x16x8xf32>) outs(%e : tensor<1x16x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.mulf %in, %c : f32
    linalg.yield %v : f32
  } -> tensor<1x16x8xf32>
  return %m : tensor<1x16x8xf32>
}

// -----

// A convolution's NHWC tensors have a leading batch of one too, and taking it
// off leaves the 4-D form every convolution matcher looks for. Measured, twelve
// models stopped compiling and six more lost half their accelerator calls.
// CHECK-LABEL: func @nhwc_keeps_its_rank
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
func.func @nhwc_keeps_its_rank(%x: tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32> {
  %c = arith.constant 2.000000e+00 : f32
  %e = tensor.empty() : tensor<1x8x8x4xf32>
  %m = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%x : tensor<1x8x8x4xf32>) outs(%e : tensor<1x8x8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.mulf %in, %c : f32
    linalg.yield %v : f32
  } -> tensor<1x8x8x4xf32>
  return %m : tensor<1x8x8x4xf32>
}

// -----

// A weight reaches a batched matmul broadcast into the batch: torch-mlir writes
// a `linalg.generic` copying a 2-D constant into a 1 x M x N one. Collapsing
// that back would leave a reshape in front of the constant, and the folder that
// turns a constant weight into an i8 one at compile time walks through
// permuting copies, not reshapes -- so the weight would be quantized again on
// every inference. Three of those is 3072 elements of an attention head's 6944.
// Reading through the broadcast leaves the constant where the folder sees it.
// CHECK-LABEL: func @weight_broadcast_into_the_batch
// CHECK:         linalg.matmul
// CHECK-SAME:      ins(%{{.*}}, %arg1 : tensor<16x32xf32>, tensor<32x8xf32>)
// CHECK-NOT:     tensor.collapse_shape %{{.*}} : tensor<1x32x8xf32>
func.func @weight_broadcast_into_the_batch(%a: tensor<1x16x32xf32>,
    %w: tensor<32x8xf32>, %init: tensor<1x16x8xf32>) -> tensor<1x16x8xf32> {
  %e = tensor.empty() : tensor<1x32x8xf32>
  %b = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%w : tensor<32x8xf32>) outs(%e : tensor<1x32x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x32x8xf32>
  %m = linalg.batch_matmul ins(%a, %b : tensor<1x16x32xf32>, tensor<1x32x8xf32>)
       outs(%init : tensor<1x16x8xf32>) -> tensor<1x16x8xf32>
  return %m : tensor<1x16x8xf32>
}
