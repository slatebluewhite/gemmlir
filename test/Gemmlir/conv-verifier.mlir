// The convolution op's arguments are checked against what the runtime accepts,
// so a call it would refuse cannot be built in the first place.

// RUN: gemmlir-opt --split-input-file --verify-diagnostics %s

// `tiled_conv_auto` compares the padding against the *undilated* kernel:
//   if (kernel_dim <= padding) {
//     printf("kernel_dim must be larger than padding\n"); exit(1); }
// A 3-tap filter at rate 4 is 9 taps wide and its shape-preserving padding is
// 4, which is past that limit. Such a border has to be materialized instead.
func.func @padding_reaches_the_kernel(%in: memref<1x24x24x3xi8>, %f: memref<3x3x3x8xi8>,
                                      %out: memref<1x24x24x8xi8>) {
  // expected-error @+1 {{padding 4 must be smaller than the 3-tap kernel}}
  gemmlir.conv2d_i8(%in, %f, %out) {dilation = 4 : i64, padding = 4 : i64}
    : (memref<1x24x24x3xi8>, memref<3x3x3x8xi8>, memref<1x24x24x8xi8>)
  return
}

// -----

// Rate 2 keeps its padding inside the kernel and is accepted.
func.func @rate_two_is_fine(%in: memref<1x24x24x3xi8>, %f: memref<3x3x3x8xi8>,
                            %out: memref<1x24x24x8xi8>) {
  gemmlir.conv2d_i8(%in, %f, %out) {dilation = 2 : i64, padding = 2 : i64}
    : (memref<1x24x24x3xi8>, memref<3x3x3x8xi8>, memref<1x24x24x8xi8>)
  return
}

// -----

// The depthwise call has the same check.
func.func @depthwise_padding(%in: memref<1x16x16x8xi8>, %f: memref<8x3x3xi8>,
                             %out: memref<1x18x18x8xi8>) {
  // expected-error @+1 {{padding 3 must be smaller than the 3-tap kernel}}
  gemmlir.depthwise_conv2d_i8(%in, %f, %out) {padding = 3 : i64}
    : (memref<1x16x16x8xi8>, memref<8x3x3xi8>, memref<1x18x18x8xi8>)
  return
}

// -----

// The runtime walks an NHWC buffer as `((n * rows + r) * cols + c) * stride`,
// with `stride` the one number it takes between two pixels: a row is that
// stride times the number of columns, and there is nowhere to say otherwise.
// A 16x16 window inside an 18x18 buffer -- which is what bufferizing a padded
// convolution makes, once the padding is on i8 -- has a wider row than that,
// and the runtime writes the wrong ones with nothing to say so. Measured, a 1x1
// convolution feeding a grouped one came back at 0.7742 relative L2, and a
// shipped model (global average pooling on a padded block) at 0.0106 where
// 0.0030 was available.
func.func @window_narrower_than_its_buffer(%in: memref<1x16x16x8xi8>, %f: memref<3x3x8x8xi8>,
    %out: memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>) {
  // expected-error @+1 {{rows must be as wide as their pixel stride says}}
  gemmlir.conv2d_i8(%in, %f, %out) {padding = 1 : i64}
    : (memref<1x16x16x8xi8>, memref<3x3x8x8xi8>,
       memref<1x16x16x8xi8, strided<[2592, 144, 8, 1], offset: 152>>)
  return
}

// -----

// A *channel* window is fine and is what a concatenation makes: the pixel
// stride widens and the rows widen with it.
func.func @channel_window_is_fine(%in: memref<1x16x16x8xi8>, %f: memref<3x3x8x8xi8>,
    %out: memref<1x16x16x8xi8, strided<[6144, 384, 24, 1], offset: 8>>) {
  gemmlir.conv2d_i8(%in, %f, %out) {padding = 1 : i64}
    : (memref<1x16x16x8xi8>, memref<3x3x8x8xi8>,
       memref<1x16x16x8xi8, strided<[6144, 384, 24, 1], offset: 8>>)
  return
}
