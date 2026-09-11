// RUN: gemmlir-opt --convert-gemmlir-to-llvm %s | FileCheck %s

// The runtime symbol is declared once at module scope with the 24-argument
// tiled_matmul_auto signature from gemmini.h (default gemmini_params.h types).
// CHECK:       llvm.func @tiled_matmul_auto(i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, f32, i32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32)
// CHECK-NOT:   llvm.func @tiled_matmul_auto

// CHECK-LABEL: @matmul_example
// A gemmini_flush(0) precedes every call: custom-3 opcode, funct7 = k_FLUSH (7).
// CHECK:         llvm.inline_asm has_side_effects is_align_stack asm_dialect = att ".insn r 0x7B, 0x3, 7, x0, x0, x0", "~{memory}"
// CHECK-NEXT:    llvm.call @tiled_matmul_auto(
// CHECK-NOT:     gemmlir.
func.func @matmul_example(%A: memref<128x128xi8>, %B: memref<128x256xi8>, %C: memref<128x256xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) : (memref<128x128xi8> x memref<128x256xi8>) -> memref<128x256xi32>
  return
}

// Two calls in one module still produce a single declaration (checked by CHECK-NOT above).
func.func @second(%A: memref<16x16xi8>, %B: memref<16x16xi8>, %C: memref<16x16xi32>) {
  gemmlir.matmul_i8(%A, %B, %C) : (memref<16x16xi8> x memref<16x16xi8>) -> memref<16x16xi32>
  return
}
