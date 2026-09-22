// linalg.matvec is the N = 1 matmul: the vectors are reshaped to single-column
// matrices, whose row stride is 1.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s
// RUN: gemmlir-opt --convert-linalg-to-gemmlir --convert-linalg-to-loops \
// RUN:   --expand-strided-metadata --lower-affine --convert-scf-to-cf \
// RUN:   --convert-gemmlir-to-llvm --convert-index-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-cf-to-llvm --convert-func-to-llvm=use-bare-ptr-memref-call-conv \
// RUN:   --reconcile-unrealized-casts --canonicalize --cse %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=IR
// RUN: not gemmlir-opt --convert-linalg-to-gemmlir %S/Inputs/matvec-i8-out.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=I8OUT

// CHECK-LABEL: func.func @mv
// CHECK-DAG:     memref.expand_shape %{{.*}} output_shape [128, 1] : memref<128xi8> into memref<128x1xi8>
// CHECK-DAG:     memref.expand_shape %{{.*}} output_shape [64, 1] : memref<64xi32> into memref<64x1xi32>
// CHECK:         gemmlir.matmul_i8({{.*}}) : (memref<64x128xi8> x memref<128x1xi8>) -> memref<64x1xi32>

//                              M       N      K        A       x       D       y
// IR-LABEL: define void @mv
// IR:         call void @tiled_matmul_auto(i64 64, i64 1, i64 128, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[Y:[0-9]+]], ptr %[[Y]],
// IR-SAME:    i64 128, i64 1, i64 1, i64 1,
func.func @mv(%A: memref<64x128xi8>, %x: memref<128xi8>, %y: memref<64xi32>) {
  linalg.matvec ins(%A, %x : memref<64x128xi8>, memref<128xi8>) outs(%y : memref<64xi32>)
  return
}

// Shapes need not be multiples of the array size; the accelerator pads, and the
// padding is not written back (checked on hardware for 37x53).
// IR-LABEL: define void @mv_odd
// IR:         call void @tiled_matmul_auto(i64 37, i64 1, i64 53,
func.func @mv_odd(%A: memref<37x53xi8>, %x: memref<53xi8>, %y: memref<37xi32>) {
  linalg.matvec ins(%A, %x : memref<37x53xi8>, memref<53xi8>) outs(%y : memref<37xi32>)
  return
}

// I8OUT: error: 'linalg.matvec' op only an i32 result can be offloaded
