// Input for batch-matmul.mlir; lit.cfg.py excludes this directory.
func.func @bad(%A: memref<4x32x64xi8>, %B: memref<4x64x48xi8>, %C: memref<4x32x48xi8>) {
  linalg.batch_matmul ins(%A, %B : memref<4x32x64xi8>, memref<4x64x48xi8>)
                      outs(%C : memref<4x32x48xi8>)
  return
}
