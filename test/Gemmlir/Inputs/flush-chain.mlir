// The @chain function of ../place-cache-flushes.mlir, on its own so the
// lowering has nothing else to legalize.

func.func @chain(%f1: memref<3x3x4x8xi8>, %f2: memref<3x3x8x8xi8>,
                 %out: memref<1x8x8x8xi8>) {
  %z = arith.constant 0 : i8
  %q = memref.alloc() : memref<1x8x8x4xi8>
  linalg.fill ins(%z : i8) outs(%q : memref<1x8x8x4xi8>)
  %t = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%q, %f1, %t) {padding = 1 : i64}
    : (memref<1x8x8x4xi8>, memref<3x3x4x8xi8>, memref<1x8x8x8xi8>)
  %u = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%t, %f2, %u) {padding = 1 : i64}
    : (memref<1x8x8x8xi8>, memref<3x3x8x8xi8>, memref<1x8x8x8xi8>)
  memref.copy %u, %out : memref<1x8x8x8xi8> to memref<1x8x8x8xi8>
  return
}
