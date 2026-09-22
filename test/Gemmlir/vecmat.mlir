// linalg.vecmat is the M = 1 matmul, the mirror of matvec: the vectors become
// single-row matrices.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s
// RUN: gemmlir-opt --convert-linalg-to-gemmlir --convert-linalg-to-loops \
// RUN:   --expand-strided-metadata --lower-affine --convert-scf-to-cf \
// RUN:   --convert-gemmlir-to-llvm --convert-index-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-cf-to-llvm --convert-func-to-llvm=use-bare-ptr-memref-call-conv \
// RUN:   --reconcile-unrealized-casts --canonicalize --cse %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=IR
// RUN: not gemmlir-opt --convert-linalg-to-gemmlir %S/Inputs/vecmat-i8-out.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=I8OUT

// CHECK-LABEL: func.func @vm
// CHECK-DAG:     memref.expand_shape %{{.*}} output_shape [1, 128] : memref<128xi8> into memref<1x128xi8>
// CHECK-DAG:     memref.expand_shape %{{.*}} output_shape [1, 64] : memref<64xi32> into memref<1x64xi32>
// CHECK:         gemmlir.matmul_i8({{.*}}) : (memref<1x128xi8> x memref<128x64xi8>) -> memref<1x64xi32>

//                              M      N       K
// IR-LABEL: define void @vm
// IR:         call void @tiled_matmul_auto(i64 1, i64 64, i64 128,
// IR-SAME:    i64 128, i64 64, i64 64, i64 64,
func.func @vm(%x: memref<128xi8>, %A: memref<128x64xi8>, %y: memref<64xi32>) {
  linalg.vecmat ins(%x, %A : memref<128xi8>, memref<128x64xi8>) outs(%y : memref<64xi32>)
  return
}

// I8OUT: error: 'linalg.vecmat' op only an i32 result can be offloaded
