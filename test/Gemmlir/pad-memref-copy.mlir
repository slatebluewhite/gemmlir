// A convolution with padding bufferizes through tensor.pad, which copies into
// the middle of a larger buffer. That copy is not a memcpy, so MLIR emits a call
// to memrefCopy -- a symbol from its C runner utils, which is a host library.
// runtime/gemmlir_rt.c provides it so a RISC-V object links.

// RUN: gemmlir-opt --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
// RUN:   --buffer-deallocation-pipeline --legalize-bare-ptr-returns \
// RUN:   --expand-strided-metadata --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm=use-bare-ptr-memref-call-conv --reconcile-unrealized-casts %s \
// RUN: | FileCheck %s

// CHECK: llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
// CHECK-LABEL: llvm.func @pad
// CHECK:         llvm.call @memrefCopy
func.func @pad(%in: tensor<2x3xf32>) -> tensor<4x5xf32> {
  %z = arith.constant 0.0 : f32
  %p = tensor.pad %in low[1, 1] high[1, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %z : f32
  } : tensor<2x3xf32> to tensor<4x5xf32>
  return %p : tensor<4x5xf32>
}
