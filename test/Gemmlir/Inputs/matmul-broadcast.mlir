// Input for transpose.mlir; lit.cfg.py excludes this directory.
func.func @bcast(%A: memref<64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi32>) {
  linalg.matmul indexing_maps = [affine_map<(m,n,k)->(k)>,
                                 affine_map<(m,n,k)->(k,n)>,
                                 affine_map<(m,n,k)->(m,n)>]
                ins(%A, %B : memref<64xi8>, memref<64x48xi8>) outs(%C : memref<32x48xi32>)
  return
}
