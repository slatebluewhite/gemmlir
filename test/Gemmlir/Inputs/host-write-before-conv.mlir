// A host write to a buffer the accelerator overwrites without reading. The
// write is dead, and on this board it is also wrong: Gemmini's writes do not
// invalidate the data cache, so the CPU reads its own line back afterwards.
func.func @filled_conv_output(%in: memref<1x16x16x3xi8>, %f: memref<3x3x3x8xi8>,
                              %out: memref<1x16x16x8xi8>) {
  %z = arith.constant 0 : i8
  linalg.fill ins(%z : i8) outs(%out : memref<1x16x16x8xi8>)
  gemmlir.conv2d_i8(%in, %f, %out) {padding = 1 : i64, scale = 2.000000e-02 : f32}
    : (memref<1x16x16x3xi8>, memref<3x3x3x8xi8>, memref<1x16x16x8xi8>)
  return
}
