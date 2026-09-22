// What gemmlir.norm_i8 refuses.

func.func @shapes_differ(%in: memref<8x16xi32>, %out: memref<8x32xi8>) {
  // expected-error @+1 {{input and output must have the same shape}}
  gemmlir.norm_i8(%in, %out) {act = #gemmlir.act<softmax>}
    : (memref<8x16xi32>, memref<8x32xi8>)
  return
}

// -----

// The runtime walks both operands with J as the row stride and takes no stride
// of its own, so a row has to be the row it thinks it is.
func.func @row_is_not_the_row(%in: memref<8x16xi32>,
    %out: memref<8x16xi8, strided<[24, 1], offset: 0>>) {
  // expected-error @+1 {{rows must be contiguous}}
  gemmlir.norm_i8(%in, %out) {act = #gemmlir.act<softmax>}
    : (memref<8x16xi32>, memref<8x16xi8, strided<[24, 1], offset: 0>>)
  return
}

// -----

// `sp_tiled_norm` branches on LAYERNORM and SOFTMAX and has no third case: an
// iGELU mvins the accumulator and never mvouts, so the output is left exactly
// as it was. Measured on the board, every element zero where the runtime's own
// `scale_and_sat` gives real values. It belongs on the matmul instead.
func.func @igelu_is_not_here(%in: memref<8x16xi32>, %out: memref<8x16xi8>) {
  // expected-error @+1 {{act must be layernorm or softmax}}
  gemmlir.norm_i8(%in, %out) {act = #gemmlir.act<igelu>}
    : (memref<8x16xi32>, memref<8x16xi8>)
  return
}

// -----

// relu belongs to the matmul's own scale pipeline, not here.
func.func @not_a_normalization(%in: memref<8x16xi32>, %out: memref<8x16xi8>) {
  // expected-error @+1 {{act must be layernorm or softmax}}
  gemmlir.norm_i8(%in, %out) {act = #gemmlir.act<relu>}
    : (memref<8x16xi32>, memref<8x16xi8>)
  return
}

// -----

// Softmax already divides each row by its own sum and multiplies by 127; the
// runtime's CPU reference substitutes that for whatever scale it was handed.
// Measured: at 0.01 the accelerator returned exactly one hundredth of the
// reference, at 1.0 the two are bit-identical.
func.func @softmax_scaled_twice(%in: memref<8x16xi32>, %out: memref<8x16xi8>) {
  // expected-error @+1 {{softmax already scales by 127 over the row's sum}}
  gemmlir.norm_i8(%in, %out) {act = #gemmlir.act<softmax>, scale = 1.000000e-02 : f32}
    : (memref<8x16xi32>, memref<8x16xi8>)
  return
}

// -----

// A convolution's accumulator holds a window of pixels, not a row, so there is
// nothing for LayerNorm or Softmax to reduce over.
func.func @conv_cannot_reduce(%in: memref<1x16x16x8xi8>, %f: memref<3x3x8x8xi8>,
                              %out: memref<1x16x16x8xi8>) {
  // expected-error @+1 {{can only fuse a pointwise activation}}
  gemmlir.conv2d_i8(%in, %f, %out) {act = #gemmlir.act<softmax>, padding = 1 : i64}
    : (memref<1x16x16x8xi8>, memref<3x3x8x8xi8>, memref<1x16x16x8xi8>)
  return
}
