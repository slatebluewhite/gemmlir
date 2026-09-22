// A matmul that does not accumulate lowers with D = NULL and full_C, so the
// runtime *writes* every element of the output. The zero fill that proved there
// was nothing to accumulate is therefore dead, and keeping it costs a store per
// output element on every inference -- 2874 of them on the two-layer CNN.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s

// CHECK-LABEL: func.func @zero_filled
// CHECK-NOT:     linalg.fill
// CHECK:         gemmlir.matmul_i8
// CHECK-SAME:      {accumulate = false}
func.func @zero_filled(%A: memref<64x64xi8>, %B: memref<64x64xi8>, %C: memref<64x64xi32>) {
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%C : memref<64x64xi32>)
  linalg.matmul ins(%A, %B : memref<64x64xi8>, memref<64x64xi8>) outs(%C : memref<64x64xi32>)
  return
}

// A non-zero fill is not dead: the matmul accumulates onto it. (Accumulating is
// the op's default, so it prints no attribute dictionary at all -- hence the
// CHECK-NEXT rather than a CHECK-SAME.)
// CHECK-LABEL: func.func @nonzero_filled
// CHECK:         linalg.fill
// CHECK:         gemmlir.matmul_i8{{.*}} -> memref<64x64xi32>
// CHECK-NEXT:    return
func.func @nonzero_filled(%A: memref<64x64xi8>, %B: memref<64x64xi8>, %C: memref<64x64xi32>) {
  %one = arith.constant 1 : i32
  linalg.fill ins(%one : i32) outs(%C : memref<64x64xi32>)
  linalg.matmul ins(%A, %B : memref<64x64xi8>, memref<64x64xi8>) outs(%C : memref<64x64xi32>)
  return
}

// Something reads the buffer between the fill and the matmul, so the zeros are
// observed and the fill stays -- and, since the buffer is no longer provably
// zero where the matmul runs, it accumulates.
// CHECK-LABEL: func.func @read_in_between
// CHECK:         linalg.fill
// CHECK:         memref.load
// CHECK:         gemmlir.matmul_i8{{.*}} -> memref<64x64xi32>
// CHECK-NEXT:    return
func.func @read_in_between(%A: memref<64x64xi8>, %B: memref<64x64xi8>, %C: memref<64x64xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %i = arith.constant 0 : index
  linalg.fill ins(%zero : i32) outs(%C : memref<64x64xi32>)
  %v = memref.load %C[%i, %i] : memref<64x64xi32>
  linalg.matmul ins(%A, %B : memref<64x64xi8>, memref<64x64xi8>) outs(%C : memref<64x64xi32>)
  return %v : i32
}

// The batch loop writes every slice, so a fill outside it is dead too.
// CHECK-LABEL: func.func @batched
// CHECK-NOT:     linalg.fill
// CHECK:         scf.for
// CHECK:           gemmlir.matmul_i8
// CHECK-SAME:        {accumulate = false}
func.func @batched(%A: memref<4x32x64xi8>, %B: memref<4x64x32xi8>, %C: memref<4x32x32xi32>) {
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%C : memref<4x32x32xi32>)
  linalg.batch_matmul ins(%A, %B : memref<4x32x64xi8>, memref<4x64x32xi8>) outs(%C : memref<4x32x32xi32>)
  return
}

// matvec is the N = 1 case and goes through the same check.
// CHECK-LABEL: func.func @vector
// CHECK-NOT:     linalg.fill
// CHECK:         gemmlir.matmul_i8
// CHECK-SAME:      {accumulate = false}
func.func @vector(%A: memref<64x64xi8>, %x: memref<64xi8>, %y: memref<64xi32>) {
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%y : memref<64xi32>)
  linalg.matvec ins(%A, %x : memref<64x64xi8>, memref<64xi8>) outs(%y : memref<64xi32>)
  return
}
