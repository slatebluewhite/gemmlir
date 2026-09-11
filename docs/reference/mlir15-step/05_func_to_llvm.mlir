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
    %13 = builtin.unrealized_conversion_cast %12 : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)> to memref<128x128xi8>
    %14 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg1, %14[0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %arg1, %15[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.insertvalue %17, %16[2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.mlir.constant(128 : index) : i64
    %20 = llvm.insertvalue %19, %18[3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.mlir.constant(256 : index) : i64
    %22 = llvm.insertvalue %21, %20[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.mlir.constant(256 : index) : i64
    %24 = llvm.insertvalue %23, %22[3, 1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.insertvalue %25, %24[4, 1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = builtin.unrealized_conversion_cast %26 : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)> to memref<128x256xi8>
    %28 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.insertvalue %arg2, %28[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.insertvalue %arg2, %29[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.mlir.constant(0 : index) : i64
    %32 = llvm.insertvalue %31, %30[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.mlir.constant(128 : index) : i64
    %34 = llvm.insertvalue %33, %32[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.mlir.constant(256 : index) : i64
    %36 = llvm.insertvalue %35, %34[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.mlir.constant(256 : index) : i64
    %38 = llvm.insertvalue %37, %36[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.mlir.constant(1 : index) : i64
    %40 = llvm.insertvalue %39, %38[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = builtin.unrealized_conversion_cast %40 : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)> to memref<128x256xi32>
    %42 = builtin.unrealized_conversion_cast %13 : memref<128x128xi8> to !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %43 = builtin.unrealized_conversion_cast %27 : memref<128x256xi8> to !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %44 = builtin.unrealized_conversion_cast %41 : memref<128x256xi32> to !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.extractvalue %42[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.extractvalue %43[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.extractvalue %44[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.mlir.null : !llvm.ptr<i8>
    %49 = llvm.mlir.constant(128 : i64) : i64
    %50 = llvm.mlir.constant(256 : i64) : i64
    %51 = llvm.mlir.constant(256 : i64) : i64
    %52 = llvm.mlir.constant(128 : i64) : i64
    %53 = llvm.mlir.constant(128 : i64) : i64
    %54 = llvm.mlir.constant(256 : i64) : i64
    %55 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %56 = llvm.mlir.constant(false) : i1
    %57 = llvm.mlir.constant(false) : i1
    %58 = llvm.mlir.constant(false) : i1
    %59 = llvm.mlir.constant(true) : i1
    %60 = llvm.mlir.constant(false) : i1
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = llvm.mlir.constant(0 : i8) : i8
    %63 = llvm.mlir.constant(0 : i32) : i32
    llvm.call @tiled_matmul_auto(%52, %54, %53, %45, %46, %48, %47, %49, %50, %50, %51, %55, %55, %55, %61, %55, %55, %56, %57, %58, %59, %60, %62, %63) : (i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i32>, i64, i64, i64, i64, f32, f32, f32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32) -> ()
    llvm.return
  }
}

