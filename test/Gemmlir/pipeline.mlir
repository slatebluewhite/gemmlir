// Full pipeline: linalg -> gemmlir -> LLVM dialect -> LLVM IR.
// RUN: gemmlir-opt --convert-linalg-to-gemmlir --convert-gemmlir-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv \
// RUN:   --reconcile-unrealized-casts --canonicalize --cse %s \
// RUN: | FileCheck %s --check-prefix=DIALECT
// RUN: gemmlir-opt --convert-linalg-to-gemmlir --convert-gemmlir-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv \
// RUN:   --reconcile-unrealized-casts --canonicalize --cse %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=IR

// With the bare-pointer calling convention the entry point takes plain pointers,
// which is what a C caller linking against the object expects.
// DIALECT: llvm.func @matmul_example(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
// DIALECT:   llvm.inline_asm
// DIALECT:   llvm.call @tiled_matmul_auto(
// DIALECT:   llvm.return

// IR: declare void @tiled_matmul_auto(i64, i64, i64, ptr, ptr, ptr, ptr, i64, i64, i64, i64, float, float, i32, i32, float, float, i1, i1, i1, i1, i1, i8, i32)
// IR: define void @matmul_example(ptr %0, ptr %1, ptr %2)
// IR:   call void asm sideeffect alignstack ".insn r 0x7B, 0x3, 7, x0, x0, x0", "~{memory}"()
//                                   dim_I    dim_J    dim_K    A       B       D       C       strides A,B,D,C (row-major: K,N,N,N)
// C is a caller-provided buffer, so linalg.matmul's `C += A*B` makes it the bias
// operand D as well.
// IR:   call void @tiled_matmul_auto(i64 128, i64 256, i64 128, ptr %0, ptr %1, ptr %2, ptr %2, i64 128, i64 256, i64 256, i64 256,
//                                   A_scale B_scale D_scale act  scale  bert_scale rep_bias trA trB full_C low_D weightA type(WS)
// IR-SAME:                          float 1.000000e+00, float 1.000000e+00, i32 1, i32 0, float 1.000000e+00, float 1.000000e+00, i1 false, i1 false, i1 false, i1 true, i1 false, i8 0, i32 1)
func.func @matmul_example(%A: memref<128x128xi8>, %B: memref<128x256xi8>, %C: memref<128x256xi32>) {
  linalg.matmul ins(%A, %B : memref<128x128xi8>, memref<128x256xi8>) outs(%C : memref<128x256xi32>)
  return
}
