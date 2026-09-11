module {
  llvm.func @tiled_matmul_auto(i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i32>, i64, i64, i64, i64, f32, f32, f32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32)
  func.func @matmul_example(%arg0: memref<128x128xi8>, %arg1: memref<128x256xi8>, %arg2: memref<128x256xi32>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<128x128xi8> to !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<128x256xi8> to !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = builtin.unrealized_conversion_cast %arg2 : memref<128x256xi32> to !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.mlir.null : !llvm.ptr<i8>
    %7 = llvm.mlir.constant(128 : i64) : i64
    %8 = llvm.mlir.constant(256 : i64) : i64
    %9 = llvm.mlir.constant(256 : i64) : i64
    %10 = llvm.mlir.constant(128 : i64) : i64
    %11 = llvm.mlir.constant(128 : i64) : i64
    %12 = llvm.mlir.constant(256 : i64) : i64
    %13 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %14 = llvm.mlir.constant(false) : i1
    %15 = llvm.mlir.constant(false) : i1
    %16 = llvm.mlir.constant(false) : i1
    %17 = llvm.mlir.constant(true) : i1
    %18 = llvm.mlir.constant(false) : i1
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(0 : i8) : i8
    %21 = llvm.mlir.constant(0 : i32) : i32
    llvm.call @tiled_matmul_auto(%10, %12, %11, %3, %4, %6, %5, %7, %8, %8, %9, %13, %13, %13, %19, %13, %13, %14, %15, %16, %17, %18, %20, %21) : (i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i32>, i64, i64, i64, i64, f32, f32, f32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32) -> ()
    return
  }
}

