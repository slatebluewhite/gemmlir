// A quantized layer comes out of a frontend adding its bias in f32, after the
// accumulator has been scaled back. The accelerator's bias operand is added to
// the i32 accumulator *before* the mvout scaling, so that bias keeps the whole
// tail of the layer in software. Since acc*s + b == (acc + b/s)*s, a constant
// bias moves inside as round(b/s), evaluated here.

// RUN: gemmlir-opt --quantize-bias-into-accumulator %s | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>
#row = affine_map<(d0, d1) -> (d0)>

// s is 0.002, so the bias becomes [127, -50]: the add is now on the i32
// accumulator and only the scaling is left in float.
// CHECK-LABEL: func.func @moves_inside
// CHECK-DAG:     %[[B:.*]] = arith.constant dense<[127, -50]> : tensor<2xi32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0, %[[B]] : tensor<4x2xi32>, tensor<2xi32>)
// CHECK-NEXT:    ^bb0(%[[A:.*]]: i32, %[[BB:.*]]: i32, %{{.*}}: i8):
// CHECK-NEXT:      %[[S:.*]] = arith.addi %[[A]], %[[BB]]
// CHECK-NEXT:      %[[F:.*]] = arith.sitofp %[[S]]
// CHECK-NEXT:      %[[M:.*]] = arith.mulf %[[F]]
// CHECK-NEXT:      arith.fptosi %[[M]]
// CHECK-NOT:       arith.addf
func.func @moves_inside(%acc: tensor<4x2xi32>) -> tensor<4x2xi8> {
  %s = arith.constant 0.002 : f32
  %bias = arith.constant dense<[0.254, -0.1]> : tensor<2xf32>
  %e = tensor.empty() : tensor<4x2xi8>
  %r = linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc, %bias : tensor<4x2xi32>, tensor<2xf32>) outs(%e : tensor<4x2xi8>) {
  ^bb0(%a: i32, %b: f32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %p = arith.addf %m, %b : f32
    %q = arith.fptosi %p : f32 to i8
    linalg.yield %q : i8
  } -> tensor<4x2xi8>
  return %r : tensor<4x2xi8>
}

// The runtime's D is a full tile or a single row repeated *down* the rows, so a
// bias broadcast over anything else can never reach it. An img2col'd
// convolution is exactly that case -- its channels are the rows -- and
// quantizing the bias there costs accuracy and buys nothing: on the NCHW CNN it
// moved the relative L2 from 0.0054 to 0.0067 without offloading one more
// operation.
// CHECK-LABEL: func.func @leaves_a_row_bias
// CHECK:         arith.addf
// CHECK-NOT:     arith.addi
func.func @leaves_a_row_bias(%acc: tensor<4x2xi32>) -> tensor<4x2xi8> {
  %s = arith.constant 0.002 : f32
  %bias = arith.constant dense<[0.254, -0.1, 0.0, 0.2]> : tensor<4xf32>
  %e = tensor.empty() : tensor<4x2xi8>
  %r = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc, %bias : tensor<4x2xi32>, tensor<4xf32>) outs(%e : tensor<4x2xi8>) {
  ^bb0(%a: i32, %b: f32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %p = arith.addf %m, %b : f32
    %q = arith.fptosi %p : f32 to i8
    linalg.yield %q : i8
  } -> tensor<4x2xi8>
  return %r : tensor<4x2xi8>
}

// When the result stays in float there is nothing to fold the bias into, and
// adding it in f32 is strictly more accurate. The last layer of a network keeps
// its own bias.
// CHECK-LABEL: func.func @leaves_a_float_result
// CHECK:         arith.addf
// CHECK-NOT:     arith.addi
func.func @leaves_a_float_result(%acc: tensor<4x2xi32>) -> tensor<4x2xf32> {
  %s = arith.constant 0.002 : f32
  %bias = arith.constant dense<[0.254, -0.1]> : tensor<2xf32>
  %e = tensor.empty() : tensor<4x2xf32>
  %r = linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc, %bias : tensor<4x2xi32>, tensor<2xf32>) outs(%e : tensor<4x2xf32>) {
  ^bb0(%a: i32, %b: f32, %o: f32):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %p = arith.addf %m, %b : f32
    linalg.yield %p : f32
  } -> tensor<4x2xf32>
  return %r : tensor<4x2xf32>
}

// A bias that is not a constant cannot be evaluated here.
// CHECK-LABEL: func.func @leaves_a_dynamic_bias
// CHECK:         arith.addf
// CHECK-NOT:     arith.addi
func.func @leaves_a_dynamic_bias(%acc: tensor<4x2xi32>, %bias: tensor<2xf32>) -> tensor<4x2xi8> {
  %s = arith.constant 0.002 : f32
  %e = tensor.empty() : tensor<4x2xi8>
  %r = linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc, %bias : tensor<4x2xi32>, tensor<2xf32>) outs(%e : tensor<4x2xi8>) {
  ^bb0(%a: i32, %b: f32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %p = arith.addf %m, %b : f32
    %q = arith.fptosi %p : f32 to i8
    linalg.yield %q : i8
  } -> tensor<4x2xi8>
  return %r : tensor<4x2xi8>
}
