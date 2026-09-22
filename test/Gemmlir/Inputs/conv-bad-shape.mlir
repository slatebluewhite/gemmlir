// Input for conv.mlir; lit.cfg.py excludes this directory.
// padding = 1 keeps a 3x3 convolution at 14x14, so a 12x12 result is wrong.
func.func @bad(%in: memref<1x14x14x16xi8>, %f: memref<3x3x16x16xi8>, %out: memref<1x12x12x16xi8>) {
  gemmlir.conv2d_i8(%in, %f, %out) {padding = 1 : i64}
      : (memref<1x14x14x16xi8>, memref<3x3x16x16xi8>, memref<1x12x12x16xi8>)
  return
}
