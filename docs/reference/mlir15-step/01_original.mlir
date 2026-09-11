module {
  func.func @matmul_example(%arg0: memref<128x128xi8>, %arg1: memref<128x256xi8>, %arg2: memref<128x256xi32>) {
    linalg.matmul ins(%arg0, %arg1 : memref<128x128xi8>, memref<128x256xi8>) outs(%arg2 : memref<128x256xi32>)
    return
  }
}

