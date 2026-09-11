module {
  func.func @matmul_example(%arg0: memref<128x128xi8>, %arg1: memref<128x256xi8>, %arg2: memref<128x256xi32>) {
    gemmlir.matmul_i8(%arg0,  %arg1,  %arg2) : (memref<128x128xi8> X memref<128x256xi8>) -> memref<128x256xi32> {transpose_lhs = false, transpose_rhs = false}
    return
  }
}

