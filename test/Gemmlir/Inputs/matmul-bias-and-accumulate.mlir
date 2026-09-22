// Input for bias.mlir; lit.cfg.py excludes this directory.
// accumulate defaults to true, and it already uses the runtime's only D pointer.
func.func @both(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %D: memref<32x48xi32>, %C: memref<32x48xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) bias(%D : memref<32x48xi32>)
    : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi32>
  return
}
