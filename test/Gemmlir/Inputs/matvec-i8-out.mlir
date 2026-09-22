// Input for matvec.mlir; lit.cfg.py excludes this directory.
func.func @bad(%A: memref<64x128xi8>, %x: memref<128xi8>, %y: memref<64xi8>) {
  linalg.matvec ins(%A, %x : memref<64x128xi8>, memref<128xi8>) outs(%y : memref<64xi8>)
  return
}
