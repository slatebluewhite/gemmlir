// RUN: gemmlir-opt --force-quantized-matmul %s | FileCheck %s
// RUN: gemmlir-opt --force-quantized-matmul %s | FileCheck %s --check-prefix=TR

// f32 tensor matmul is rewritten into qcast -> i8 storage -> i8xi8->i32 matmul -> dcast.
// CHECK-LABEL: func.func @matmul_f32
// Each cast goes where its operand is defined -- so that one value quantized
// at one scale is one loop however many contractions read it -- and the two
// come out in the order the arguments were defined, not the order the matmul
// reads them.
// CHECK-DAG:     %[[QA:.*]] = quant.qcast %arg0 : tensor<128x128xf32> to tensor<128x128x!quant.uniform<i8:f32, 2.000000e-02>>
// CHECK-DAG:     %[[QB:.*]] = quant.qcast %arg1 : tensor<128x256xf32> to tensor<128x256x!quant.uniform<i8:f32, 2.000000e-02>>
// CHECK-DAG:     %[[A8:.*]] = quant.scast %[[QA]] : {{.*}} to tensor<128x128xi8>
// CHECK-DAG:     %[[B8:.*]] = quant.scast %[[QB]] : {{.*}} to tensor<128x256xi8>
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<128x256xi32>
// CHECK:         %[[Z:.*]] = linalg.fill ins(%{{.*}} : i32) outs(%[[E]] : tensor<128x256xi32>)
// CHECK:         %[[M:.*]] = linalg.matmul ins(%[[A8]], %[[B8]] : tensor<128x128xi8>, tensor<128x256xi8>) outs(%[[Z]] : tensor<128x256xi32>)
// CHECK:         %[[QM:.*]] = quant.scast %[[M]] : tensor<128x256xi32> to tensor<128x256x!quant.uniform<i32:f32, 4.000000e-04>>
// CHECK:         %[[R:.*]] = quant.dcast %[[QM]] : {{.*}} to tensor<128x256xf32>
// linalg.matmul accumulates into its output operand, and %arg2 is not known to
// be zero, so the incoming values are added back after dequantizing. A frontend
// really does put things there -- torch-mlir broadcasts a convolution's bias
// into the init tensor -- and dropping it cost a PyTorch CNN a relative L2 error
// of 0.33 where 0.012 was available.
// CHECK:         %[[S:.*]] = arith.addf %[[R]], %arg2
// CHECK:         return %[[S]]
func.func @matmul_f32(%A: tensor<128x128xf32>, %B: tensor<128x256xf32>, %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x256xf32>) outs(%C : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// A zero fill needs no such addition.
// CHECK-LABEL: func.func @matmul_zeroed
// CHECK:         quant.dcast
// CHECK-NOT:     arith.addf
// CHECK:         return
func.func @matmul_zeroed(%A: tensor<128x128xf32>, %B: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x256xf32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<128x256xf32>) -> tensor<128x256xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x256xf32>) outs(%f : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// Nothing in the rewrite is matmul-specific, so a convolution goes the same
// way: i8 operands into the i32 accumulator the named op extends into, which is
// the form --convert-linalg-to-gemmlir folds into conv2d_i8. The weight is a
// constant, so it quantizes at compile time.
// CHECK-LABEL: func.func @conv_f32
// CHECK-DAG:     arith.constant dense<{{.*}}> : tensor<1x1x2x2xi8>
// CHECK:         %[[QA:.*]] = quant.qcast %arg0
// CHECK:         %[[A8:.*]] = quant.scast %[[QA]] : {{.*}} to tensor<1x4x4x2xi8>
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      dilations = dense<1>
// CHECK-SAME:      strides = dense<2>
// CHECK-SAME:      ins(%[[A8]], %{{.*}} : tensor<1x4x4x2xi8>, tensor<1x1x2x2xi8>)
// CHECK-SAME:      -> tensor<1x2x2x2xi32>
func.func @conv_f32(%in: tensor<1x4x4x2xf32>) -> tensor<1x2x2x2xf32> {
  %w = arith.constant dense<[[[[0.254, -0.1], [0.0, 0.2]]]]> : tensor<1x1x2x2xf32>
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x2x2x2xf32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
       ins(%in, %w : tensor<1x4x4x2xf32>, tensor<1x1x2x2xf32>) outs(%f : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %c : tensor<1x2x2x2xf32>
}

// MLIR 22 says "this operand is transposed" by overriding indexing_maps, so the
// quantized matmul has to carry them over. Defaulting them would quietly
// compute the untransposed product.
// TR-DAG:   #[[MB:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// TR-LABEL: func.func @transposed_rhs
// TR:         linalg.matmul indexing_maps = [#{{.*}}, #[[MB]], #{{.*}}]
// TR-SAME:      -> tensor<8x8xi32>
#a = affine_map<(m, n, k) -> (m, k)>
#bt = affine_map<(m, n, k) -> (n, k)>
#c = affine_map<(m, n, k) -> (m, n)>
func.func @transposed_rhs(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<8x8xf32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
  %r = linalg.matmul indexing_maps = [#a, #bt, #c]
       ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>) outs(%f : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// A batch of contractions goes the same way, one scale for all of them. It is
// what a global average pool over a batch of images becomes:
// `--average-pool-to-contraction` writes `ones(N, 1, H*W) x image(N, H*W, C)`,
// because collapsing the batch into the contraction would sum across images.
// CHECK-LABEL: func.func @batch_matmul_f32
// CHECK-DAG:     %[[A8:.*]] = quant.scast %{{.*}} to tensor<4x1x256xi8>
// CHECK-DAG:     %[[B8:.*]] = quant.scast %{{.*}} to tensor<4x256x8xi8>
// CHECK:         %[[Z:.*]] = linalg.fill ins(%{{.*}} : i32) outs(%{{.*}} : tensor<4x1x8xi32>)
// CHECK:         linalg.batch_matmul ins(%[[A8]], %[[B8]] : tensor<4x1x256xi8>, tensor<4x256x8xi8>)
// CHECK-SAME:      outs(%[[Z]] : tensor<4x1x8xi32>)
// CHECK:         quant.dcast
func.func @batch_matmul_f32(%a: tensor<4x1x256xf32>, %b: tensor<4x256x8xf32>)
    -> tensor<4x1x8xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<4x1x8xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<4x1x8xf32>) -> tensor<4x1x8xf32>
  %m = linalg.batch_matmul ins(%a, %b : tensor<4x1x256xf32>, tensor<4x256x8xf32>)
       outs(%f : tensor<4x1x8xf32>) -> tensor<4x1x8xf32>
  return %m : tensor<4x1x8xf32>
}

// -----

// `--split-grouped-conv` gives every group a slice of one filter, so the
// weights reach the contraction behind a `tensor.extract_slice`. Looking for
// the constant has to go through it and read only that window: the first half
// of this constant reaches 8.0 and the second only 0.5, and each group's scale
// is its own half's.
//
// Failing to find it at all is what used to happen, and the fallback was the
// activation's scale -- for a filter next to an activation that is about three
// of the 256 levels, and it came back as a ResNeXt block at 0.1295 relative L2
// where the same convolution alone is 0.0096.
// CHECK-LABEL: func.func @weights_behind_a_slice
// CHECK:         quant.qcast %extracted_slice
// CHECK-SAME:      to tensor<4x2x!quant.uniform<i8:f32, 0.062992125984251968>>
// CHECK:         quant.qcast %extracted_slice_0
// CHECK-SAME:      to tensor<4x2x!quant.uniform<i8:f32, 0.003937007874015748>>
func.func @weights_behind_a_slice(%a: tensor<3x4xf32>, %init: tensor<3x2xf32>)
    -> (tensor<3x2xf32>, tensor<3x2xf32>) {
  %w = arith.constant dense<[[8.0, -3.0, 0.5, -0.25],
                            [1.0,  2.0, 0.5,  0.125],
                            [-7.0, 4.0, 0.25, 0.5],
                            [6.0, -2.0, 0.5, -0.5]]> : tensor<4x4xf32>
  %left = tensor.extract_slice %w[0, 0] [4, 2] [1, 1] : tensor<4x4xf32> to tensor<4x2xf32>
  %right = tensor.extract_slice %w[0, 2] [4, 2] [1, 1] : tensor<4x4xf32> to tensor<4x2xf32>
  %l = linalg.matmul {gemmlir.activation_scale = 2.000000e-02 : f64}
       ins(%a, %left : tensor<3x4xf32>, tensor<4x2xf32>)
       outs(%init : tensor<3x2xf32>) -> tensor<3x2xf32>
  %r = linalg.matmul {gemmlir.activation_scale = 2.000000e-02 : f64}
       ins(%a, %right : tensor<3x4xf32>, tensor<4x2xf32>)
       outs(%init : tensor<3x2xf32>) -> tensor<3x2xf32>
  return %l, %r : tensor<3x2xf32>, tensor<3x2xf32>
}

// -----

// A transformer's two busiest contractions multiply an activation by an
// activation: `Q @ K.T` and `probs @ V` are the `@` operator, not a layer, and
// there is no weight whose range can be read. Taking the left operand's scale
// for the right one is not close -- attention probabilities live in [0, 0.13]
// and the values they weigh reach 2.8, so quantizing the values at the
// probabilities' scale flattens them. Measured on one head, 0.8352 relative L2
// against 0.0115 once both were measured.
//
// `gemmlir.rhs_activation_scale` is what a calibration puts there.
// CHECK-LABEL: func.func @activation_times_activation
// CHECK-DAG:     quant.qcast %arg0 : tensor<8x16xf32> to tensor<8x16x!quant.uniform<i8:f32, 2.000000e-02>>
// CHECK-DAG:     quant.qcast %arg1 : tensor<16x8xf32> to tensor<16x8x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @activation_times_activation(%a: tensor<8x16xf32>, %b: tensor<16x8xf32>,
                                       %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %m = linalg.matmul {gemmlir.activation_scale = 2.000000e-02 : f64,
                      gemmlir.rhs_activation_scale = 5.000000e-01 : f64}
       ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>)
       outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %m : tensor<8x8xf32>
}

// -----

// A constant right operand keeps its own range: that is exact where a
// calibration is a sample, so the annotation does not override it. The weight
// folds to an i8 constant and leaves no cast of its own, so the accumulator's
// scale is what says which range won -- 0.02 * 8/127, not 0.02 * 0.5.
// CHECK-LABEL: func.func @constant_beats_the_annotation
// CHECK:         quant.scast %{{.*}} : tensor<3x2xi32> to
// CHECK-SAME:      !quant.uniform<i32:f32, 0.0012598425196850393>
func.func @constant_beats_the_annotation(%a: tensor<3x4xf32>, %init: tensor<3x2xf32>)
    -> tensor<3x2xf32> {
  %w = arith.constant dense<[[8.0, -3.0], [1.0, 2.0], [-7.0, 4.0], [6.0, -2.0]]>
     : tensor<4x2xf32>
  %m = linalg.matmul {gemmlir.activation_scale = 2.000000e-02 : f64,
                      gemmlir.rhs_activation_scale = 5.000000e-01 : f64}
       ins(%a, %w : tensor<3x4xf32>, tensor<4x2xf32>)
       outs(%init : tensor<3x2xf32>) -> tensor<3x2xf32>
  return %m : tensor<3x2xf32>
}

// -----

// A weight that reaches the contraction **broadcast** is folded anyway.
//
// `linalg.batch_matmul` takes a weight of the same rank as the activation, so a
// frontend copies the 2-D constant once per batch element -- and the
// quantization then lands on the copies, which means the weight is converted on
// every inference. ConvNeXt's MLP is this shape: a 768 x 3072 weight converted
// twice per run, 81.4 million elements of its 83.2 million of scalar work.
//
// The walk up to the constant used to require every step to be a **permutation**
// of its input; a broadcast reads through a map that drops a dimension, so it
// was refused. The evaluation below already unravels the destination index and
// maps it into the source, so a dropped dimension needs nothing extra there --
// only the *write* has to stay a permutation, or an element would be folded
// more than once.
//
// The constant comes out at the contraction's shape, so a batch of two puts two
// copies of it in the binary. That is the deliberate half of the trade, and it
// was measured: folding at the weight's own shape and spreading it at run time
// halves the constant and costs **5.28 ms against 1.42** on a probe with a
// 48 x 192 and a 192 x 48 weight, because the spread is a transposing copy --
// torch-mlir writes such a weight as `(b, i, j) -> (j, i)`, broadcasting and
// transposing in one step.
//
// Measured: **9.37 to 1.42 ms**, byte-identical to the previous object's output
// and 0 of 40 against the CPU reference.
// CHECK-LABEL: func.func @a_broadcast_weight_folds
// CHECK:         %[[W:.*]] = arith.constant dense<{{.*}}> : tensor<2x4x8xi8>
// CHECK:         linalg.batch_matmul
// CHECK-SAME:      %[[W]] : tensor<2x3x4xi8>, tensor<2x4x8xi8>
// Nothing converts it at run time.
// CHECK-NOT:     arith.trunci
func.func @a_broadcast_weight_folds(%a: tensor<2x3x4xf32>) -> tensor<2x3x8xf32> {
  %zero = arith.constant 0.0 : f32
  %w = arith.constant dense<[[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
                             [0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6],
                             [0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9],
                             [1.0, -1.1, 1.2, -1.3, 1.4, -1.5, 1.6, -1.7]]>
      : tensor<4x8xf32>
  %we = tensor.empty() : tensor<2x4x8xf32>
  %wb = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                        iterator_types = ["parallel","parallel","parallel"]}
    ins(%w : tensor<4x8xf32>) outs(%we : tensor<2x4x8xf32>) {
  ^bb0(%v: f32, %out: f32):
    linalg.yield %v : f32
  } -> tensor<2x4x8xf32>
  %e = tensor.empty() : tensor<2x3x8xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  %m = linalg.batch_matmul ins(%a, %wb : tensor<2x3x4xf32>, tensor<2x4x8xf32>)
       outs(%f : tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  return %m : tensor<2x3x8xf32>
}
