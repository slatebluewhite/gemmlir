// A batch of images through the whole quantized pipeline. The runtime takes a
// batch size for a convolution and a row count for a matmul, so nothing about
// the mapping changes -- this is here because several of the rewrites around it
// had to learn that the leading dimension is not always one.

// RUN: gemmlir-opt --average-pool-to-contraction --canonicalize \
// RUN:   --fold-batch-norm --canonicalize --force-quantized-matmul --canonicalize \
// RUN:   --lower-quant-ops --round-quantized-casts --strip-func-quant-types --canonicalize \
// RUN:   --convert-elementwise-to-linalg --canonicalize \
// RUN:   --fuse-elementwise-around-matmul --canonicalize \
// RUN:   --quantize-bias-into-accumulator --canonicalize \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
// RUN:   --buffer-deallocation-pipeline --convert-linalg-to-gemmlir %s \
// RUN: | FileCheck %s

#chan = affine_map<(d0, d1, d2, d3) -> (d3)>
#img = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// The batch reaches the call: the runtime's first argument is `batch_size`, and
// `--convert-gemmlir-to-llvm` reads it off the operand's shape.
// CHECK-LABEL: func.func @batched
// CHECK:         gemmlir.conv2d_i8
// CHECK-SAME:      (memref<4x8x8x4xi8>, memref<3x3x4x4xi8>, memref<4x6x6x4xi8>)

// The global average pool over a batch keeps its batch dimension, so it is a
// batch_matmul -- which lowers to one call per image rather than one call.
// CHECK:         scf.for
// CHECK:           gemmlir.matmul_i8
// CHECK-SAME:        (memref<1x36xi8, {{.*}}> x memref<36x4xi8, {{.*}}>) -> memref<1x4xi32
func.func @batched(%in: tensor<4x8x8x4xf32>) -> tensor<4x1x1x4xf32> {
  %zero = arith.constant 0.0 : f32
  %count = arith.constant 3.600000e+01 : f32
  %w = arith.constant dense<0.1> : tensor<3x3x4x4xf32>
  %bias = arith.constant dense<[0.5, -0.25, 0.125, 0.0]> : tensor<4xf32>

  %e = tensor.empty() : tensor<4x6x6x4xf32>
  %init = linalg.generic {indexing_maps = [#chan, #img],
                          iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%bias : tensor<4xf32>) outs(%e : tensor<4x6x6x4xf32>) {
  ^bb0(%b: f32, %o: f32):
    linalg.yield %b : f32
  } -> tensor<4x6x6x4xf32>
  %c = linalg.conv_2d_nhwc_hwcf {gemmlir.activation_scale = 2.000000e-02 : f64,
                                 dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : tensor<4x8x8x4xf32>, tensor<3x3x4x4xf32>)
    outs(%init : tensor<4x6x6x4xf32>) -> tensor<4x6x6x4xf32>
  %r = linalg.generic {indexing_maps = [#img, #img],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<4x6x6x4xf32>) outs(%e : tensor<4x6x6x4xf32>) {
  ^bb0(%x: f32, %o: f32):
    %g = arith.cmpf ugt, %x, %zero : f32
    %s = arith.select %g, %x, %zero : f32
    linalg.yield %s : f32
  } -> tensor<4x6x6x4xf32>

  %win = tensor.empty() : tensor<6x6xf32>
  %po = tensor.empty() : tensor<4x1x1x4xf32>
  %pf = linalg.fill ins(%zero : f32) outs(%po : tensor<4x1x1x4xf32>) -> tensor<4x1x1x4xf32>
  %p = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%r, %win : tensor<4x6x6x4xf32>, tensor<6x6xf32>)
    outs(%pf : tensor<4x1x1x4xf32>) -> tensor<4x1x1x4xf32>
  %avg = linalg.generic {indexing_maps = [#img, #img],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%p : tensor<4x1x1x4xf32>) outs(%po : tensor<4x1x1x4xf32>) {
  ^bb0(%x: f32, %o: f32):
    %d = arith.divf %x, %count : f32
    linalg.yield %d : f32
  } -> tensor<4x1x1x4xf32>
  return %avg : tensor<4x1x1x4xf32>
}
