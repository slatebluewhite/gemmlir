// RUN: gemmlir-opt %s --set-target-data-layout --split-input-file | FileCheck %s
// RUN: gemmlir-opt %s --set-target-data-layout="data-layout= target-triple=" --split-input-file | FileCheck %s --check-prefix=NONE

// Without these the translation emits no `target datalayout`, LLVM's default
// applies, and an i64 is claimed to be four-byte aligned.

// CHECK:      module attributes {
// CHECK-SAME:   llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
// CHECK-SAME:   llvm.target_triple = "riscv64-unknown-linux-gnu"
// An empty option leaves the module alone.
// NONE:      module {
// NONE-NOT:    llvm.data_layout
module {
  func.func @f(%a: memref<64xi64>, %i: index) -> i64 {
    %v = memref.load %a[%i] : memref<64xi64>
    return %v : i64
  }
}

// -----

// An attribute the module already carries is replaced, not appended to.

// CHECK:      module attributes {
// CHECK-SAME:   llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
// CHECK-NOT:    llvm.data_layout = "E-m:e"
// ...and what the module already carried is kept when the option is empty.
// NONE:       llvm.data_layout = "E-m:e"
module attributes {llvm.data_layout = "E-m:e"} {
  func.func @g() {
    return
  }
}
