module attributes {llvm.data_layout = ""} {
  llvm.func @tiled_matmul_auto(i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i32>, i64, i64, i64, i64, f32, f32, f32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32)
  llvm.func @matmul_example(%arg0: !llvm.ptr<i8>, %arg1: !llvm.ptr<i8>, %arg2: !llvm.ptr<i32>) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.mlir.constant(128 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.mlir.constant(128 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg1, %13[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg1, %14[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.mlir.constant(128 : index) : i64
    %19 = llvm.insertvalue %18, %17[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.mlir.constant(256 : index) : i64
    %21 = llvm.insertvalue %20, %19[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.mlir.constant(256 : index) : i64
    %23 = llvm.insertvalue %22, %21[3, 1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.insertvalue %24, %23[4, 1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.insertvalue %arg2, %26[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.insertvalue %arg2, %27[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.mlir.constant(128 : index) : i64
    %32 = llvm.insertvalue %31, %30[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.mlir.constant(256 : index) : i64
    %34 = llvm.insertvalue %33, %32[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.mlir.constant(256 : index) : i64
    %36 = llvm.insertvalue %35, %34[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.insertvalue %37, %36[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %42 = llvm.mlir.null : !llvm.ptr<i8>
    %43 = llvm.mlir.constant(128 : i64) : i64
    %44 = llvm.mlir.constant(256 : i64) : i64
    %45 = llvm.mlir.constant(256 : i64) : i64
    %46 = llvm.mlir.constant(128 : i64) : i64
    %47 = llvm.mlir.constant(128 : i64) : i64
    %48 = llvm.mlir.constant(256 : i64) : i64
    %49 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %50 = llvm.mlir.constant(false) : i1
    %51 = llvm.mlir.constant(false) : i1
    %52 = llvm.mlir.constant(false) : i1
    %53 = llvm.mlir.constant(true) : i1
    %54 = llvm.mlir.constant(false) : i1
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = llvm.mlir.constant(0 : i8) : i8
    %57 = llvm.mlir.constant(0 : i32) : i32
    llvm.call @tiled_matmul_auto(%46, %48, %47, %39, %40, %42, %41, %43, %44, %44, %45, %49, %49, %49, %55, %49, %49, %50, %51, %52, %53, %54, %56, %57) : (i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i32>, i64, i64, i64, i64, f32, f32, f32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32) -> ()
    llvm.return
  }
}

