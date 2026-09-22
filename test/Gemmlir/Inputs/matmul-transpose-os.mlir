// Input for transpose.mlir; lit.cfg.py excludes this directory.
func.func @os_transpose(%A: memref<64x32xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) : (memref<64x32xi8> x memref<64x48xi8>) -> memref<32x48xi32>
    {transpose_lhs = true, dataflow = #gemmlir.dataflow<os>}
  return
}
