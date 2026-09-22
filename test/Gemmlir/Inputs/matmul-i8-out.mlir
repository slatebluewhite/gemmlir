// Input for linalg-to-gemmlir.mlir; lit.cfg.py excludes this directory.
func.func @matmul_i8_out(%A: memref<64x64xi8>, %B: memref<64x64xi8>, %C: memref<64x64xi8>) {
  linalg.matmul ins(%A, %B : memref<64x64xi8>, memref<64x64xi8>) outs(%C : memref<64x64xi8>)
  return
}
