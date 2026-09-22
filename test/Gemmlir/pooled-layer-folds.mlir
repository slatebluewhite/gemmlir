// Where `--average-pool-to-contraction` has to sit in the quantized pipeline,
// and what it is worth.
//
// A frontend sums an average pool in f32 and divides afterwards, so the
// convolution feeding it has no i8 result to fold into: the convolution *and*
// the pool both stay scalar loops, and a scalar convolution is the most
// expensive thing this compiler can leave behind. Turning the pool into a
// depthwise convolution first makes it an ordinary layer -- calibrated,
// quantized and folded like any other -- and the convolution above it gets its
// requantization back.
//
// It has to run **before** quantization, which is why it is placed next to
// --fold-batch-norm rather than later.
//
// Measured on the board: `apb` one offloaded operation of five, and 42.8 ms.

// RUN: gemmlir-opt --force-quantized-matmul --canonicalize \
// RUN:   --share-branch-quantization --canonicalize \
// RUN:   --lower-quant-ops --round-quantized-casts --strip-func-quant-types --canonicalize \
// RUN:   --convert-elementwise-to-linalg --canonicalize \
// RUN:   --fuse-elementwise-around-matmul --canonicalize \
// RUN:   --requantize-before-pooling --canonicalize \
// RUN:   --fuse-elementwise-around-matmul --canonicalize \
// RUN:   --quantize-bias-into-accumulator --canonicalize \
// RUN:   --one-shot-bufferize=bufferize-function-boundaries=1 \
// RUN:   --buffer-deallocation-pipeline --convert-linalg-to-gemmlir \
// RUN:   %S/Inputs/conv-avgpool-linear.mlir | FileCheck %s --check-prefix=WITHOUT

// RUN: gemmlir-opt --average-pool-to-contraction --canonicalize \
// RUN:   --force-quantized-matmul --canonicalize \
// RUN:   --share-branch-quantization --canonicalize \
// RUN:   --lower-quant-ops --round-quantized-casts --strip-func-quant-types --canonicalize \
// RUN:   --convert-elementwise-to-linalg --canonicalize \
// RUN:   --fuse-elementwise-around-matmul --canonicalize \
// RUN:   --requantize-before-pooling --canonicalize \
// RUN:   --fuse-elementwise-around-matmul --canonicalize \
// RUN:   --quantize-bias-into-accumulator --canonicalize \
// RUN:   --one-shot-bufferize=bufferize-function-boundaries=1 \
// RUN:   --buffer-deallocation-pipeline --convert-linalg-to-gemmlir \
// RUN:   %S/Inputs/conv-avgpool-linear.mlir | FileCheck %s --check-prefix=WITH

// Left as a pool, only the linear layer reaches the accelerator and the
// convolution is a scalar loop.
// WITHOUT: linalg.conv_2d_nhwc_hwcf
// WITHOUT: linalg.pooling_nhwc_sum
// WITHOUT: gemmlir.matmul_i8(
// WITHOUT-NOT: gemmlir.conv2d_i8(

// Turned into a contraction, all three layers are accelerator calls and nothing
// is left in software.
// WITH:      gemmlir.conv2d_i8(
// WITH-SAME:   bias(
// WITH:      gemmlir.depthwise_conv2d_i8(
// WITH:      gemmlir.matmul_i8(
// WITH-NOT:  linalg.conv_2d_nhwc_hwcf
// WITH-NOT:  linalg.pooling
