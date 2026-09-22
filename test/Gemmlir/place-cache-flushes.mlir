// Gemmini reads through the L2 and does not probe this board's L1, in either
// direction, so `gemmlir_rt.c` displaces the L1 on both sides of every call --
// about 0.12 ms a call. Between two convolutions of a fused network the host
// touches nothing, so almost none of those flushes can matter.

// RUN: gemmlir-opt --place-cache-flushes --split-input-file %s | FileCheck %s
// RUN: gemmlir-opt --place-cache-flushes --convert-gemmlir-to-llvm \
// RUN:   %S/Inputs/flush-chain.mlir | FileCheck %s --check-prefix=LLVM

// A quantize, two convolutions, a dequantize: one flush, before the first call.
// It does double duty -- it drops the lines the host wrote into the first
// convolution's input, and the lines it left behind reading the last one's
// output on the *previous* call, which is why nothing needs a flush after.
// CHECK-LABEL: func @chain
// CHECK:         gemmlir.conv2d_i8(%alloc, %arg0, %alloc_0)
// CHECK-SAME:      {gemmlir.no_flush_after, padding = 1 : i64}
// CHECK:         gemmlir.conv2d_i8(%alloc_0, %arg1, %alloc_1)
// CHECK-SAME:      {gemmlir.no_flush_after, gemmlir.no_flush_before, padding = 1 : i64}

// LLVM-LABEL: func @chain
// LLVM:         llvm.call @gemmlir_flush()
// LLVM:         llvm.call @tiled_conv_stride_auto
// LLVM-NOT:     llvm.call @gemmlir_flush()
// LLVM:         llvm.call @tiled_conv_stride_auto
// LLVM-NOT:     llvm.call @gemmlir_flush()
func.func @chain(%f1: memref<3x3x4x8xi8>, %f2: memref<3x3x8x8xi8>,
                 %out: memref<1x8x8x8xi8>) {
  %z = arith.constant 0 : i8
  %q = memref.alloc() : memref<1x8x8x4xi8>
  linalg.fill ins(%z : i8) outs(%q : memref<1x8x8x4xi8>)
  %t = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%q, %f1, %t) {padding = 1 : i64}
    : (memref<1x8x8x4xi8>, memref<3x3x4x8xi8>, memref<1x8x8x8xi8>)
  %u = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%t, %f2, %u) {padding = 1 : i64}
    : (memref<1x8x8x8xi8>, memref<3x3x8x8xi8>, memref<1x8x8x8xi8>)
  memref.copy %u, %out : memref<1x8x8x8xi8> to memref<1x8x8x8xi8>
  return
}

// -----

// A border the host materializes between the two calls -- what a dilated
// convolution's padding leaves behind when the runtime refuses to fold it --
// is a buffer the accelerator reads and the host just wrote, so the second
// call keeps its flush.
// CHECK-LABEL: func @host_builds_an_input
// CHECK:         gemmlir.conv2d_i8(%alloc, %arg0, %alloc_0)
// CHECK-SAME:      {gemmlir.no_flush_after, padding = 1 : i64}
// CHECK:         gemmlir.conv2d_i8(%alloc_1, %arg1, %alloc_2)
// CHECK-SAME:      {gemmlir.no_flush_after}
func.func @host_builds_an_input(%f1: memref<3x3x4x8xi8>, %f2: memref<3x3x8x8xi8>,
                                %out: memref<1x8x8x8xi8>) {
  %z = arith.constant 0 : i8
  %q = memref.alloc() : memref<1x8x8x4xi8>
  linalg.fill ins(%z : i8) outs(%q : memref<1x8x8x4xi8>)
  %t = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%q, %f1, %t) {padding = 1 : i64}
    : (memref<1x8x8x4xi8>, memref<3x3x4x8xi8>, memref<1x8x8x8xi8>)
  %p = memref.alloc() : memref<1x10x10x8xi8>
  linalg.fill ins(%z : i8) outs(%p : memref<1x10x10x8xi8>)
  %w = memref.subview %p[0, 1, 1, 0] [1, 8, 8, 8] [1, 1, 1, 1]
     : memref<1x10x10x8xi8> to memref<1x8x8x8xi8, strided<[800, 80, 8, 1], offset: 88>>
  memref.copy %t, %w : memref<1x8x8x8xi8> to memref<1x8x8x8xi8, strided<[800, 80, 8, 1], offset: 88>>
  %u = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%p, %f2, %u) {padding = 0 : i64}
    : (memref<1x10x10x8xi8>, memref<3x3x8x8xi8>, memref<1x8x8x8xi8>)
  memref.copy %u, %out : memref<1x8x8x8xi8> to memref<1x8x8x8xi8>
  return
}

// -----

// An operation whose memory effects this pass cannot see is not an excuse to
// guess: every flush stays.
// CHECK-LABEL: func @opaque_call
// CHECK-NOT:     gemmlir.no_flush
func.func private @something(memref<1x8x8x8xi8>)
func.func @opaque_call(%f1: memref<3x3x4x8xi8>, %f2: memref<3x3x8x8xi8>,
                       %out: memref<1x8x8x8xi8>) {
  %z = arith.constant 0 : i8
  %q = memref.alloc() : memref<1x8x8x4xi8>
  linalg.fill ins(%z : i8) outs(%q : memref<1x8x8x4xi8>)
  %t = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%q, %f1, %t) {padding = 1 : i64}
    : (memref<1x8x8x4xi8>, memref<3x3x4x8xi8>, memref<1x8x8x8xi8>)
  func.call @something(%t) : (memref<1x8x8x8xi8>) -> ()
  %u = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%t, %f2, %u) {padding = 1 : i64}
    : (memref<1x8x8x8xi8>, memref<3x3x8x8xi8>, memref<1x8x8x8xi8>)
  memref.copy %u, %out : memref<1x8x8x8xi8> to memref<1x8x8x8xi8>
  return
}

// -----

// An accelerator call reads its operands **on the host** too.
//
// `gemmlir_first_read` in the runtime reads a byte a page of every operand
// before the call, so the accelerator does not meet a page with no PTE and read
// it as zeros. That is a host read like any other: it pulls lines in, and if a
// call has since overwritten that buffer, the lines it pulls are the stale ones.
//
// Leaving it out of the simulation is what let this pass take a needed flush
// away. The symptom was an answer that depended on which program had run
// *before* -- a large model first is what makes the host hold old lines for the
// address at all -- about one run in four, and it vanished under any
// instrumentation, because reading the buffer to check it is itself the missing
// read.
//
// Here the second call reads what the first wrote, so the flush between them
// stays.
// A **local** buffer, so nothing outside marks it resident: a call writes it, a
// call reads it -- and that read is the runtime's own operand touch, which is
// what puts the host's lines there -- a call overwrites it, and the host reads
// it. The flush after the overwrite is the one that drops those lines, and
// without the operand touch in the model nothing said it was needed.
// (A label for the default prefix too, so the earlier CHECK-NOT stops here
// rather than running to the end of the file.)
// CHECK-LABEL: func @a_call_leaves_lines_behind
// SECOND-LABEL: func @a_call_leaves_lines_behind
// SECOND:         gemmlir.resadd_i8
// SECOND-SAME:      {gemmlir.no_flush_after}
// SECOND:         gemmlir.matmul_i8
// The overwrite keeps the flush after it: only `no_flush_before` is on it.
// SECOND:         gemmlir.resadd_i8
// SECOND-SAME:      {gemmlir.no_flush_before}
// SECOND:         memref.copy

// RUN: gemmlir-opt --place-cache-flushes %s | FileCheck %s --check-prefix=SECOND
func.func @a_call_leaves_lines_behind(%a: memref<8x8xi8>, %b: memref<8x8xi8>,
                                      %w: memref<8x8xi8>, %out: memref<8x8xi32>,
                                      %dst: memref<8x8xi8>) {
  %t = memref.alloc() : memref<8x8xi8>
  gemmlir.resadd_i8(%a, %b, %t) : (memref<8x8xi8> x memref<8x8xi8>) -> memref<8x8xi8>
  gemmlir.matmul_i8(%t, %w, %out) : (memref<8x8xi8> x memref<8x8xi8>) -> memref<8x8xi32>
  gemmlir.resadd_i8(%b, %a, %t) : (memref<8x8xi8> x memref<8x8xi8>) -> memref<8x8xi8>
  memref.copy %t, %dst : memref<8x8xi8> to memref<8x8xi8>
  memref.dealloc %t : memref<8x8xi8>
  return
}

// -----

// A batch matmul folds per slice, which leaves an `scf.for` at the top level
// with an accelerator call inside it. `scf.for` declares no memory effects of
// its own, so this used to end the analysis and keep every flush in the
// function -- a transformer is 24 such loops, and `vit_tiny` kept all 124 of
// its flushes.
//
// The loop is modelled as one host step over everything inside it, and the call
// **inside** keeps both flushes, which is what makes that sound. The two
// convolutions after it are then free to drop theirs.

// CHECK-LABEL: func @sliced
// CHECK:         scf.for
// CHECK:           gemmlir.matmul_i8
// CHECK-NOT:       gemmlir.no_flush
// CHECK:         gemmlir.conv2d_i8(
// CHECK-SAME:      {gemmlir.no_flush_after, padding = 1 : i64}
// CHECK:         gemmlir.conv2d_i8(
// CHECK-SAME:      {gemmlir.no_flush_after, gemmlir.no_flush_before, padding = 1 : i64}
func.func @sliced(%a: memref<4x8x8xi8>, %b: memref<4x8x8xi8>,
                  %f1: memref<3x3x4x8xi8>, %f2: memref<3x3x8x8xi8>,
                  %seed: memref<1x8x8x4xi8>, %out: memref<1x8x8x8xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %acc = memref.alloc() : memref<4x8x8xi32>
  scf.for %i = %c0 to %c4 step %c1 {
    %sa = memref.subview %a[%i, 0, 0] [1, 8, 8] [1, 1, 1]
      : memref<4x8x8xi8> to memref<8x8xi8, strided<[8, 1], offset: ?>>
    %sb = memref.subview %b[%i, 0, 0] [1, 8, 8] [1, 1, 1]
      : memref<4x8x8xi8> to memref<8x8xi8, strided<[8, 1], offset: ?>>
    %sc = memref.subview %acc[%i, 0, 0] [1, 8, 8] [1, 1, 1]
      : memref<4x8x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>
    gemmlir.matmul_i8(%sa, %sb, %sc)
      : (memref<8x8xi8, strided<[8, 1], offset: ?>> x memref<8x8xi8, strided<[8, 1], offset: ?>>)
      -> memref<8x8xi32, strided<[8, 1], offset: ?>>
  }
  %t = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%seed, %f1, %t) {padding = 1 : i64}
    : (memref<1x8x8x4xi8>, memref<3x3x4x8xi8>, memref<1x8x8x8xi8>)
  %u = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%t, %f2, %u) {padding = 1 : i64}
    : (memref<1x8x8x8xi8>, memref<3x3x8x8xi8>, memref<1x8x8x8xi8>)
  memref.copy %u, %out : memref<1x8x8x8xi8> to memref<1x8x8x8xi8>
  return
}
