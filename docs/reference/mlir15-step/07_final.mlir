module attributes {llvm.data_layout = ""} {
  llvm.func @tiled_matmul_auto(i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i32>, i64, i64, i64, i64, f32, f32, f32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32)
  llvm.func @matmul_example(%arg0: !llvm.ptr<i8>, %arg1: !llvm.ptr<i8>, %arg2: !llvm.ptr<i32>) {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(true) : i1
    %3 = llvm.mlir.constant(false) : i1
    %4 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(256 : i64) : i64
    %6 = llvm.mlir.constant(128 : i64) : i64
    %7 = llvm.mlir.null : !llvm.ptr<i8>
    llvm.call @tiled_matmul_auto(%6, %5, %6, %arg0, %arg1, %7, %arg2, %6, %5, %5, %5, %4, %4, %4, %1, %4, %4, %3, %3, %3, %2, %3, %0, %1) : (i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i32>, i64, i64, i64, i64, f32, f32, f32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32) -> ()
    llvm.return
  }
}

