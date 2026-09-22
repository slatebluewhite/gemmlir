// RUN: gemmlir-opt --split-input-file --raise-spatial-sum-to-pool %s | FileCheck %s

// `x.mean(dim=(2, 3))` on NCHW: a sum over both spatial axes, then a divide.
// The sum is a global pool and the rest of the pipeline only knows it under
// that name.

#img = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: func @mean_nchw
// CHECK:         %[[W:.*]] = tensor.empty() : tensor<8x8xf32>
// CHECK:         %[[P:.*]] = linalg.pooling_nchw_sum
// CHECK-SAME:      ins(%{{.*}}, %[[W]] : tensor<1x32x8x8xf32>, tensor<8x8xf32>)
// CHECK-SAME:      -> tensor<1x32x1x1xf32>
// CHECK:         tensor.collapse_shape %[[P]] {{\[}}[0], [1, 2, 3]]
// CHECK-SAME:      tensor<1x32x1x1xf32> into tensor<1x32xf32>
func.func @mean_nchw(%x: tensor<1x32x8x8xf32>) -> tensor<1x32xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<1x32xf32>) -> tensor<1x32xf32>
  %s = linalg.generic {indexing_maps = [#img, #chan],
                       iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%x : tensor<1x32x8x8xf32>) outs(%init : tensor<1x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<1x32xf32>
  return %s : tensor<1x32xf32>
}

// -----

// The same written in NHWC -- which axes are summed is the only thing that
// tells the two apart.

#img = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d0, d3)>

// CHECK-LABEL: func @mean_nhwc
// CHECK:         %[[P:.*]] = linalg.pooling_nhwc_sum
// CHECK-SAME:      -> tensor<1x1x1x16xf32>
// CHECK:         tensor.collapse_shape %[[P]] {{\[}}[0, 1, 2], [3]]
func.func @mean_nhwc(%x: tensor<1x4x4x16xf32>) -> tensor<1x16xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<1x16xf32>) -> tensor<1x16xf32>
  %s = linalg.generic {indexing_maps = [#img, #chan],
                       iterator_types = ["parallel", "reduction", "reduction", "parallel"]}
      ins(%x : tensor<1x4x4x16xf32>) outs(%init : tensor<1x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %out, %in : f32
    linalg.yield %a : f32
  } -> tensor<1x16xf32>
  return %s : tensor<1x16xf32>
}

// -----

// An accumulator that does not start at zero is not a sum of the image.

#img = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: func @not_from_zero
// CHECK-NOT:     linalg.pooling
// CHECK:         linalg.generic
func.func @not_from_zero(%x: tensor<1x32x8x8xf32>) -> tensor<1x32xf32> {
  %one = arith.constant 1.0 : f32
  %e = tensor.empty() : tensor<1x32xf32>
  %init = linalg.fill ins(%one : f32) outs(%e : tensor<1x32xf32>) -> tensor<1x32xf32>
  %s = linalg.generic {indexing_maps = [#img, #chan],
                       iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%x : tensor<1x32x8x8xf32>) outs(%init : tensor<1x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<1x32xf32>
  return %s : tensor<1x32xf32>
}

// -----

// A body that does anything besides accumulate keeps its loop: this one relus
// on the way in, which is a different operation and not one the pool has.

#img = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: func @relu_then_sum
// CHECK-NOT:     linalg.pooling
// CHECK:         linalg.generic
func.func @relu_then_sum(%x: tensor<1x32x8x8xf32>) -> tensor<1x32xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<1x32xf32>) -> tensor<1x32xf32>
  %s = linalg.generic {indexing_maps = [#img, #chan],
                       iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%x : tensor<1x32x8x8xf32>) outs(%init : tensor<1x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c = arith.maximumf %in, %zero : f32
    %a = arith.addf %c, %out : f32
    linalg.yield %a : f32
  } -> tensor<1x32xf32>
  return %s : tensor<1x32xf32>
}

// -----

// Summing one axis is a different reduction -- a pool reduces both.

#img = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#keep = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func @one_axis
// CHECK-NOT:     linalg.pooling
// CHECK:         linalg.generic
func.func @one_axis(%x: tensor<1x32x8x8xf32>) -> tensor<1x32x8xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x32x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<1x32x8xf32>) -> tensor<1x32x8xf32>
  %s = linalg.generic {indexing_maps = [#img, #keep],
                       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%x : tensor<1x32x8x8xf32>) outs(%init : tensor<1x32x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %a = arith.addf %in, %out : f32
    linalg.yield %a : f32
  } -> tensor<1x32x8xf32>
  return %s : tensor<1x32x8xf32>
}
