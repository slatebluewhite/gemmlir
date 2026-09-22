// The accelerator's matmul is 2-D, so linalg.batch_matmul becomes an scf.for
// over rank-reduced subviews. Those carry a dynamic offset, which the lowering
// folds into the pointer -- using alignedPtr alone would drop it.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s
// RUN: gemmlir-opt --convert-linalg-to-gemmlir --convert-linalg-to-loops \
// RUN:   --expand-strided-metadata --lower-affine --convert-scf-to-cf \
// RUN:   --convert-gemmlir-to-llvm --convert-index-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-cf-to-llvm --convert-func-to-llvm=use-bare-ptr-memref-call-conv \
// RUN:   --reconcile-unrealized-casts --canonicalize --cse %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=IR
// RUN: not gemmlir-opt --convert-linalg-to-gemmlir %S/Inputs/batch-matmul-i8-out.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=I8OUT

// CHECK-LABEL: func.func @bmm
// CHECK:         scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step
// CHECK:           memref.subview %{{.*}}[%[[IV]], 0, 0] [1, 32, 64] [1, 1, 1]
// CHECK-SAME:      to memref<32x64xi8, strided<[64, 1], offset: ?>>
// CHECK:           gemmlir.matmul_i8

// The row strides are the slices' own (64, 48, 48), and the batch offset reaches
// the runtime through the pointers rather than the strides.
// IR-LABEL: define void @bmm
// IR:         getelementptr
// IR:         call void @tiled_matmul_auto(i64 32, i64 48, i64 64,
// IR-SAME:    i64 64, i64 48, i64 48, i64 48,
func.func @bmm(%A: memref<4x32x64xi8>, %B: memref<4x64x48xi8>, %C: memref<4x32x48xi32>) {
  linalg.batch_matmul ins(%A, %B : memref<4x32x64xi8>, memref<4x64x48xi8>)
                      outs(%C : memref<4x32x48xi32>)
  return
}

// A slice of a wider buffer keeps that buffer's row stride, not its own shape.
// IR-LABEL: define void @sliced
// IR:         call void @tiled_matmul_auto(i64 32, i64 48, i64 64,
// IR-SAME:    i64 64, i64 48, i64 96, i64 96,
func.func @sliced(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %big: memref<64x96xi32>) {
  %s = memref.subview %big[16, 32] [32, 48] [1, 1]
     : memref<64x96xi32> to memref<32x48xi32, strided<[96, 1], offset: 1568>>
  gemmlir.matmul_i8(%A, %B, %s) : (memref<32x64xi8> x memref<64x48xi8>)
    -> memref<32x48xi32, strided<[96, 1], offset: 1568>> {accumulate = false}
  return
}

// I8OUT: error: 'linalg.batch_matmul' op only an i32 result can be offloaded
