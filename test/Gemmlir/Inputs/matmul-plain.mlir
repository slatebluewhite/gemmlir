// Input for mvin-scale.mlir; lit.cfg.py excludes this directory.
func.func @plain(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi32>) {
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>) outs(%C : memref<32x48xi32>)
  return
}
