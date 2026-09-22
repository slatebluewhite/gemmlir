// The Gemmini dataflow is a pass option on --convert-linalg-to-gemmlir that lands
// as an attribute on the produced op. Its numbering is gemmini.h's
// `enum tiled_matmul_type_t {OS, WS, CPU}`.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s --check-prefix=DEFAULT
// RUN: gemmlir-opt --convert-linalg-to-gemmlir="dataflow=os" %s | FileCheck %s --check-prefix=OS
// RUN: gemmlir-opt --convert-linalg-to-gemmlir --convert-gemmlir-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv --reconcile-unrealized-casts %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=IRWS
// RUN: gemmlir-opt --convert-linalg-to-gemmlir="dataflow=os" --convert-gemmlir-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv --reconcile-unrealized-casts %s \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=IROS
// RUN: not gemmlir-opt --convert-linalg-to-gemmlir="dataflow=nonsense" %s 2>&1 | FileCheck %s --check-prefix=BAD

// ws is the default, and a default-valued attribute is elided when printed.
// DEFAULT-LABEL: func.func @matmul
// DEFAULT:         gemmlir.matmul_i8(%arg0, %arg1, %arg2)
// DEFAULT-NOT:     dataflow

// OS: gemmlir.matmul_i8(%arg0, %arg1, %arg2) {{.*}} {dataflow = #gemmlir.dataflow<os>}

// The attribute reaches the runtime call as the trailing i32 argument.
// IRWS: call void @tiled_matmul_auto(
// IRWS-SAME: i8 0, i32 1)
// IROS: call void @tiled_matmul_auto(
// IROS-SAME: i8 0, i32 0)

// BAD: error: unknown dataflow 'nonsense', expected one of: os, ws, cpu

func.func @matmul(%A: memref<128x128xi8>, %B: memref<128x256xi8>, %C: memref<128x256xi32>) {
  linalg.matmul ins(%A, %B : memref<128x128xi8>, memref<128x256xi8>) outs(%C : memref<128x256xi32>)
  return
}
