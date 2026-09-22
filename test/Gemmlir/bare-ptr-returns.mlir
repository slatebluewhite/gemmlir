// Under the bare-pointer convention a function returning a memref hands back one
// pointer, and --convert-func-to-llvm returns the *allocated* one. An aligned
// allocation therefore starts up to alignment-1 bytes after it.

// RUN: gemmlir-opt --legalize-bare-ptr-returns %s | FileCheck %s
// RUN: gemmlir-opt --legalize-bare-ptr-returns --expand-strided-metadata \
// RUN:   --finalize-memref-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv \
// RUN:   --reconcile-unrealized-casts %s | mlir-translate --mlir-to-llvmir \
// RUN: | FileCheck %s --check-prefix=IR

// The returned allocation loses its alignment...
// CHECK-LABEL: func.func @returned
// CHECK:         memref.alloc()
// CHECK-NOT:     alignment
// CHECK:         return
func.func @returned() -> memref<8xf32> {
  %a = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
  return %a : memref<8xf32>
}

// ...so there is no round-up between the allocation and the return: the pointer
// the caller gets is the one the data is at.
// IR-LABEL: define ptr @returned()
// IR:         call ptr @malloc(i64 32)
// IR-NOT:     urem
// IR:         ret ptr

// A buffer that stays inside the function keeps its alignment: those are the
// ones the accelerator reads, and nothing returns their pointer.
// CHECK-LABEL: func.func @internal
// CHECK:         memref.alloc() {alignment = 64 : i64}
func.func @internal(%v: f32) {
  %a = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
  %c0 = arith.constant 0 : index
  memref.store %v, %a[%c0] : memref<8xf32>
  memref.dealloc %a : memref<8xf32>
  return
}

// The allocation is not always the returned value itself. A layer whose result
// is reshaped on the way out returns a view of the buffer, which carries the
// same two pointers -- so the alignment still has to go. By the time this pass
// runs, --expand-strided-metadata has turned the reshape into a
// `memref.reinterpret_cast`, and missing that is not a small error: the whole
// result reads from the wrong address, which took a quantized convolution
// block to a relative L2 of 0.94 against its own f32 reference.
// CHECK-LABEL: func.func @returned_through_a_view
// CHECK:         memref.alloc()
// CHECK-NOT:     alignment
// CHECK:         return
func.func @returned_through_a_view() -> memref<1x2x4xf32> {
  %a = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
  %v = memref.expand_shape %a [[0, 1], [2]] output_shape [1, 2, 4]
       : memref<2x4xf32> into memref<1x2x4xf32>
  return %v : memref<1x2x4xf32>
}

// CHECK-LABEL: func.func @returned_through_a_reinterpret
// CHECK:         memref.alloc()
// CHECK-NOT:     alignment
// CHECK:         return
func.func @returned_through_a_reinterpret() -> memref<1x2x4xf32> {
  %a = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
  %v = memref.reinterpret_cast %a to offset: [0], sizes: [1, 2, 4], strides: [8, 4, 1]
       : memref<2x4xf32> to memref<1x2x4xf32>
  return %v : memref<1x2x4xf32>
}

// An offset is not something dropping an alignment could fix, so a view that
// starts somewhere else keeps the allocation as it is rather than pretending.
// CHECK-LABEL: func.func @returned_with_an_offset
// CHECK:         memref.alloc() {alignment = 64 : i64}
func.func @returned_with_an_offset() -> memref<4xf32, strided<[1], offset: 4>> {
  %a = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
  %v = memref.reinterpret_cast %a to offset: [4], sizes: [4], strides: [1]
       : memref<2x4xf32> to memref<4xf32, strided<[1], offset: 4>>
  return %v : memref<4xf32, strided<[1], offset: 4>>
}
