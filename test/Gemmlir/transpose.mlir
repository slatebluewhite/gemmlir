// MLIR 22 spells a transposed matmul as an indexing_maps override on
// linalg.matmul. Ignoring it would compute the untransposed product, so the
// maps are read -- and then the match is refused, because this board's
// accelerator does not compute a transposed operand.
//
// `tiled_matmul_auto` takes the flags and the runtime's own CPU implementation
// honours them exactly. The accelerator does not: measured against a plain-C
// reference, transposing B came back 2045 of 2048 elements wrong and
// transposing A 2048 of 2048, the same on every call, where the same object on
// the CPU runtime was exact. So a transposed operand stays a loop, and the
// lowering refuses the flags as well in case anything else builds them.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s
// RUN: not gemmlir-opt --convert-linalg-to-gemmlir --convert-gemmlir-to-llvm %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=REFUSED
// RUN: not gemmlir-opt --convert-linalg-to-gemmlir %S/Inputs/matmul-broadcast.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=BCAST
// RUN: not gemmlir-opt --convert-gemmlir-to-llvm %S/Inputs/matmul-transpose-os.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=OS
// RUN: not gemmlir-opt --convert-gemmlir-to-llvm %S/Inputs/matmul-transpose-both.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=BOTH

// A is stored (K, M) = 64x32, so the product is 32x48 with K = 64.
// CHECK-LABEL: func.func @t_a
// CHECK:         gemmlir.matmul_i8({{.*}}) {{.*}} {transpose_lhs = true}

// ...and the lowering refuses it, so nothing wrong is ever compiled.
// REFUSED: does not compute a transposed operand
func.func @t_a(%A: memref<64x32xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi32>) {
  linalg.matmul indexing_maps = [affine_map<(m,n,k)->(k,m)>,
                                 affine_map<(m,n,k)->(k,n)>,
                                 affine_map<(m,n,k)->(m,n)>]
                ins(%A, %B : memref<64x32xi8>, memref<64x48xi8>) outs(%C : memref<32x48xi32>)
  return
}

// B stored (N, K) = 48x64.
// CHECK-LABEL: func.func @t_b
// CHECK:         gemmlir.matmul_i8({{.*}}) {{.*}} {transpose_rhs = true}
func.func @t_b(%A: memref<32x64xi8>, %B: memref<48x64xi8>, %C: memref<32x48xi32>) {
  linalg.matmul indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                                 affine_map<(m,n,k)->(n,k)>,
                                 affine_map<(m,n,k)->(m,n)>]
                ins(%A, %B : memref<32x64xi8>, memref<48x64xi8>) outs(%C : memref<32x48xi32>)
  return
}

// A broadcast map is not a transpose and must not be silently dropped.
// BCAST: error: 'linalg.matmul' op indexing_maps are not a plain or transposed matmul

// The runtime refuses these two outright, so catch them here.
// OS: error: 'gemmlir.matmul_i8' op the 'os' dataflow cannot transpose an operand
// BOTH: error: 'gemmlir.matmul_i8' op 'ws' can transpose one operand but not both
