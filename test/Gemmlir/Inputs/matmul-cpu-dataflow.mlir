// Input for dataflow-attr.mlir; lit.cfg.py excludes this directory.
func.func @cpu_dataflow(%A: memref<16x16xi8>, %B: memref<16x16xi8>, %C: memref<16x16xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) : (memref<16x16xi8> x memref<16x16xi8>) -> memref<16x16xi32> {dataflow = #gemmlir.dataflow<cpu>}
  return
}
