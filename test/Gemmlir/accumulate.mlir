// linalg.matmul means C += A*B. The lowering gets that by handing C to the
// runtime as the bias operand D as well (verified on hardware that D may alias
// C); when C is provably zero it passes NULL instead and saves reading it back.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir \
// RUN:   --convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata \
// RUN:   --convert-gemmlir-to-llvm --convert-index-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv \
// RUN:   --reconcile-unrealized-casts --canonicalize --cse %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s

// C arrives from the caller: its contents matter, so D = C and stride_D = stride_C.
// CHECK-LABEL: define void @accumulating
// CHECK:         call void @tiled_matmul_auto(i64 64, i64 64, i64 64, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[C:[0-9]+]], ptr %[[C]],
func.func @accumulating(%A: memref<64x64xi8>, %B: memref<64x64xi8>, %C: memref<64x64xi32>) {
  linalg.matmul ins(%A, %B : memref<64x64xi8>, memref<64x64xi8>) outs(%C : memref<64x64xi32>)
  return
}

// A zero fill immediately before proves there is nothing to accumulate.
// CHECK-LABEL: define void @zero_filled
// CHECK:         call void @tiled_matmul_auto(i64 64, i64 64, i64 64, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr null,
func.func @zero_filled(%A: memref<64x64xi8>, %B: memref<64x64xi8>, %C: memref<64x64xi32>) {
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%C : memref<64x64xi32>)
  linalg.matmul ins(%A, %B : memref<64x64xi8>, memref<64x64xi8>) outs(%C : memref<64x64xi32>)
  return
}

// A non-zero fill proves nothing.
// CHECK-LABEL: define void @nonzero_filled
// CHECK:         call void @tiled_matmul_auto(i64 64, i64 64, i64 64, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[C2:[0-9]+]], ptr %[[C2]],
func.func @nonzero_filled(%A: memref<64x64xi8>, %B: memref<64x64xi8>, %C: memref<64x64xi32>) {
  %one = arith.constant 1 : i32
  linalg.fill ins(%one : i32) outs(%C : memref<64x64xi32>)
  linalg.matmul ins(%A, %B : memref<64x64xi8>, memref<64x64xi8>) outs(%C : memref<64x64xi32>)
  return
}
