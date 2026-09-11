module {
  func.func @matmul_example(%A: memref<128x128xi8>, %B: memref<128x256xi8>, %C: memref<128x256xi32>) {
    linalg.matmul ins(%A, %B : memref<128x128xi8>, memref<128x256xi8>) outs(%C : memref<128x256xi32>)
    return
  }
}
