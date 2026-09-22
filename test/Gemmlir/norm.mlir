// The normalization unit: LayerNorm and the I-BERT softmax, applied in the
// accumulator's scale pipeline on the way out. It needs a bitstream whose
// Gemmini was built with `norms = true` -- the U280 board carries one as of
// 2026-09-12 (a `Normalizer` module and 637 `igelu` references in the generated
// Verilog). Both are bit-identical to the runtime's own CPU formulation on the
// board; see docs/pipeline.md.
//
// The normalization runs along a row, so the input is the i32 accumulator a
// matmul left behind and the output is the i8 the next layer reads.

// RUN: gemmlir-opt %s | gemmlir-opt | FileCheck %s
// RUN: gemmlir-opt --convert-gemmlir-to-llvm %s | FileCheck %s --check-prefix=LLVM
// RUN: gemmlir-opt --split-input-file --verify-diagnostics %S/Inputs/norm-invalid.mlir

// CHECK-LABEL: func @softmax
// CHECK:         gemmlir.norm_i8(%arg0, %arg1)
// CHECK-SAME:      {act = #gemmlir.act<softmax>}
// CHECK-SAME:      : (memref<32x64xi32>, memref<32x64xi8>)

// The runtime takes I, J, the two pointers, the scale, the activation code and
// the type. 4 is SOFTMAX in gemmini.h and in Chisel's Activation.scala alike.
// LLVM-LABEL: func @softmax
// LLVM:         llvm.call @gemmlir_flush()
// LLVM:         llvm.call @tiled_norm_auto
// LLVM-SAME:      (i64, i64, !llvm.ptr, !llvm.ptr, f32, i32, i32) -> ()
func.func @softmax(%in: memref<32x64xi32>, %out: memref<32x64xi8>) {
  gemmlir.norm_i8(%in, %out) {act = #gemmlir.act<softmax>, scale = 1.000000e+00 : f32}
    : (memref<32x64xi32>, memref<32x64xi8>)
  return
}

// LayerNorm does take a scale: it centres and divides by the row's standard
// deviation, and what that lands on is the caller's to say.
// CHECK-LABEL: func @layernorm
// CHECK:         gemmlir.norm_i8(%arg0, %arg1)
// CHECK-SAME:      {act = #gemmlir.act<layernorm>, scale = 2.500000e-01 : f32}
func.func @layernorm(%in: memref<8x16xi32>, %out: memref<8x16xi8>) {
  gemmlir.norm_i8(%in, %out) {act = #gemmlir.act<layernorm>, scale = 2.500000e-01 : f32}
    : (memref<8x16xi32>, memref<8x16xi8>)
  return
}

// -----

// The matmul's own scale pipeline takes them too, and there iGELU works --
// `tiled_matmul` configures its constants where `tiled_norm` does not. All
// three are bit-identical to the runtime's CPU path on the board.
// CHECK-LABEL: func @matmul_igelu
// CHECK:         gemmlir.matmul_i8_scale
// CHECK-SAME:      act = #gemmlir.act<igelu>
func.func @matmul_igelu(%a: memref<32x64xi8>, %b: memref<64x64xi8>, %c: memref<32x64xi8>) {
  gemmlir.matmul_i8_scale(%a, %b, %c) : (memref<32x64xi8> x memref<64x64xi8>) -> memref<32x64xi8>
    {act = #gemmlir.act<igelu>, scale = 1.000000e-02 : f32}
  return
}

// `bert_scale` is what one unit of the accumulator is worth, and it is what
// makes this a softmax rather than the exponent of a row of integers: the
// runtime derives qln2, qb and qc from it. For a quantized matmul the right
// value is `lhs_scale * rhs_scale`. Measured against float softmax of the exact
// product: told, every element within 0.5 of 127; not told, 123.8 of 127.
// CHECK-LABEL: func @matmul_softmax
// CHECK:         gemmlir.matmul_i8_scale
// CHECK-SAME:      act = #gemmlir.act<softmax>
// CHECK-SAME:      bert_scale = 6.250000e-04 : f32
// CHECK-SAME:      lhs_scale = 2.500000e-02 : f32
func.func @matmul_softmax(%a: memref<32x64xi8>, %b: memref<64x64xi8>, %c: memref<32x64xi8>) {
  gemmlir.matmul_i8_scale(%a, %b, %c) : (memref<32x64xi8> x memref<64x64xi8>) -> memref<32x64xi8>
    {act = #gemmlir.act<softmax>, scale = 1.000000e+00 : f32,
     lhs_scale = 2.500000e-02 : f32, rhs_scale = 2.500000e-02 : f32,
     bert_scale = 6.250000e-04 : f32}
  return
}
