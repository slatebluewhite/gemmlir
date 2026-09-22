// Input for vecmat.mlir; lit.cfg.py excludes this directory.
func.func @bad(%x: memref<128xi8>, %A: memref<128x64xi8>, %y: memref<64xi8>) {
  linalg.vecmat ins(%x, %A : memref<128xi8>, memref<128x64xi8>) outs(%y : memref<64xi8>)
  return
}
