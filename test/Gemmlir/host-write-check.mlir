// Gemmini's writes do not invalidate this board's data cache, so a line the CPU
// wrote survives the accelerator overwriting the memory underneath it and the
// CPU reads its own stale data. Measured directly, with no compiler involved:
// filling the output buffer before the call left 43 to 69 of 784 elements
// reading back wrong, six rounds running; evicting the cache between the call
// and the read, or never touching the buffer at all, gave 0 of 784, six rounds
// running. There is no instruction to do it properly with -- the board is
// rv64imafdc, no Zicbom.
//
// Only a write the accelerator then overwrites *without reading* is rejected,
// because such a write is dead anyway. An accumulating matmul reads its output
// as the bias, so a write there is the accumulator's starting value.

// RUN: not gemmlir-opt --convert-linalg-to-gemmlir %S/Inputs/host-write-before-conv.mlir 2>&1 | FileCheck %s
// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s --check-prefix=OK

// CHECK-DAG: error: the host writes this operation's output buffer beforehand
// CHECK-DAG: remark: the write is here

// Accumulating reads the output as the bias, so the fill belongs there.
// OK-LABEL: func.func @accumulating_is_fine
// OK:         linalg.fill
// OK:         gemmlir.matmul_i8
func.func @accumulating_is_fine(%a: memref<8x27xi8>, %b: memref<27x256xi8>,
                                %c: memref<8x256xi32>) {
  %one = arith.constant 1 : i32
  linalg.fill ins(%one : i32) outs(%c : memref<8x256xi32>)
  gemmlir.matmul_i8(%a, %b, %c) : (memref<8x27xi8> x memref<27x256xi8>) -> memref<8x256xi32>
  return
}

// Writing the accelerator's *input* is the ordinary case and is fine: the
// hazard is about reading back what the accelerator wrote.
// OK-LABEL: func.func @writing_an_input_is_fine
// OK:         linalg.fill
// OK:         gemmlir.matmul_i8
func.func @writing_an_input_is_fine(%b: memref<27x256xi8>, %c: memref<8x256xi32>) {
  %z = arith.constant 0 : i8
  %a = memref.alloc() : memref<8x27xi8>
  linalg.fill ins(%z : i8) outs(%a : memref<8x27xi8>)
  gemmlir.matmul_i8(%a, %b, %c) : (memref<8x27xi8> x memref<27x256xi8>) -> memref<8x256xi32> {accumulate = false}
  memref.dealloc %a : memref<8x27xi8>
  return
}
