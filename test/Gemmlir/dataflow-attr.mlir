// A dataflow attribute written by hand on the op is honoured by the lowering,
// and `cpu` is refused because this op lowers with full_C.

// RUN: gemmlir-opt --convert-gemmlir-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv --reconcile-unrealized-casts %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s
// RUN: not gemmlir-opt --convert-gemmlir-to-llvm %S/Inputs/matmul-cpu-dataflow.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=CPU

// CHECK-LABEL: define void @explicit_os
// CHECK:         call void asm sideeffect
// CHECK:         call void @tiled_matmul_auto(
// CHECK-SAME:    i8 0, i32 0)
func.func @explicit_os(%A: memref<16x16xi8>, %B: memref<16x16xi8>, %C: memref<16x16xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) : (memref<16x16xi8> x memref<16x16xi8>) -> memref<16x16xi32> {dataflow = #gemmlir.dataflow<os>}
  return
}

// CPU: error: 'gemmlir.matmul_i8' op dataflow 'cpu' is not available here
