// Attention is the one shape where **both** operands of a contraction are
// activations, twice over, with a softmax between the two -- and the heads are
// the batch dimension, so each is a `linalg.batch_matmul`. torchvision's
// VisionTransformer is 74 contractions and 24 of them are these two; the
// remaining 50 are ordinary projections.
//
// Neither of the two has a constant to read a range off, so both need
// `gemmlir.rhs_activation_scale`, and the calibration has to have seen them:
// `nn.MultiheadAttention` reaches them through `F.multi_head_attention_forward`,
// which answers `has_torch_function`, so a `TorchFunctionMode` is handed the
// whole call and everything inside it runs with the mode switched off.

// RUN: gemmlir-opt --force-quantized-matmul --canonicalize \
// RUN:   --lower-quant-ops --round-quantized-casts --strip-func-quant-types --canonicalize \
// RUN:   --convert-elementwise-to-linalg --canonicalize \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
// RUN:   --buffer-deallocation-pipeline --convert-linalg-to-gemmlir %s \
// RUN: | FileCheck %s

#id3  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#row3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Both contractions offload, one loop over the heads each, and the softmax
// between them stops neither.
// CHECK-LABEL: func.func @attention_head
// CHECK:         scf.for
// CHECK:           gemmlir.matmul_i8
// CHECK:         math.exp
// CHECK:         scf.for
// CHECK:           gemmlir.matmul_i8
func.func @attention_head(%q: tensor<3x8x4xf32>, %k: tensor<3x4x8xf32>,
                          %v: tensor<3x8x4xf32>) -> tensor<3x8x4xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %se = tensor.empty() : tensor<3x8x8xf32>
  %si = linalg.fill ins(%zero : f32) outs(%se : tensor<3x8x8xf32>) -> tensor<3x8x8xf32>
  %scores = linalg.batch_matmul {gemmlir.activation_scale = 2.000000e-02 : f64,
                                 gemmlir.rhs_activation_scale = 3.000000e-02 : f64}
      ins(%q, %k : tensor<3x8x4xf32>, tensor<3x4x8xf32>)
      outs(%si : tensor<3x8x8xf32>) -> tensor<3x8x8xf32>

  // softmax, written out: an exponential, a row sum, a divide
  %ee = tensor.empty() : tensor<3x8x8xf32>
  %ex = linalg.generic {indexing_maps = [#id3, #id3],
                        iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%scores : tensor<3x8x8xf32>) outs(%ee : tensor<3x8x8xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<3x8x8xf32>
  %re = tensor.empty() : tensor<3x8xf32>
  %ri = linalg.fill ins(%zero : f32) outs(%re : tensor<3x8xf32>) -> tensor<3x8xf32>
  %sum = linalg.generic {indexing_maps = [#id3, #row3],
                         iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%ex : tensor<3x8x8xf32>) outs(%ri : tensor<3x8xf32>) {
  ^bb0(%in: f32, %o: f32):
    %t = arith.addf %in, %o : f32
    linalg.yield %t : f32
  } -> tensor<3x8xf32>
  %pe = tensor.empty() : tensor<3x8x8xf32>
  %probs = linalg.generic {indexing_maps = [#id3, #row3, #id3],
                           iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ex, %sum : tensor<3x8x8xf32>, tensor<3x8xf32>) outs(%pe : tensor<3x8x8xf32>) {
  ^bb0(%in: f32, %s: f32, %o: f32):
    %d = arith.divf %in, %s : f32
    linalg.yield %d : f32
  } -> tensor<3x8x8xf32>

  %oe = tensor.empty() : tensor<3x8x4xf32>
  %oi = linalg.fill ins(%zero : f32) outs(%oe : tensor<3x8x4xf32>) -> tensor<3x8x4xf32>
  %out = linalg.batch_matmul {gemmlir.activation_scale = 1.000000e-02 : f64,
                              gemmlir.rhs_activation_scale = 4.000000e-02 : f64}
      ins(%probs, %v : tensor<3x8x8xf32>, tensor<3x8x4xf32>)
      outs(%oi : tensor<3x8x4xf32>) -> tensor<3x8x4xf32>
  return %out : tensor<3x8x4xf32>
}
