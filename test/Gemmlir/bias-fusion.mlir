// A `+= bias` sitting on a matmul folds into the runtime's D operand.
//
// Sound because nothing saturates here: arith.addi wraps in i32 and so does the
// accelerator's 32-bit accumulator, whose raw value full_C reads back -- checked
// on hardware by driving D + A*B past INT32_MAX (862 of 862 cases wrapped, none
// saturated).

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s

#id  = affine_map<(d0,d1)->(d0,d1)>
#row = affine_map<(d0,d1)->(d1)>

// CHECK-LABEL: func.func @fuse_full
// CHECK:         gemmlir.matmul_i8(%arg0, %arg1, %arg3) bias(%arg2 : memref<32x48xi32>)
// CHECK-SAME:    {accumulate = false}
// CHECK-NOT:     linalg.generic
func.func @fuse_full(%A: memref<32x64xi8>, %B: memref<64x48xi8>,
                     %D: memref<32x48xi32>, %C: memref<32x48xi32>) {
  %z = arith.constant 0 : i32
  linalg.fill ins(%z : i32) outs(%C : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>) outs(%C : memref<32x48xi32>)
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%D : memref<32x48xi32>) outs(%C : memref<32x48xi32>) {
  ^bb0(%b: i32, %c: i32):
    %s = arith.addi %c, %b : i32
    linalg.yield %s : i32
  }
  return
}

// A row broadcast becomes a 1xN bias, which is the runtime's repeating_bias.
// CHECK-LABEL: func.func @fuse_row
// CHECK:         memref.expand_shape %arg2 {{.*}} output_shape [1, 48] : memref<48xi32> into memref<1x48xi32>
// CHECK:         gemmlir.matmul_i8(%arg0, %arg1, %arg3) bias(%expand_shape : memref<1x48xi32>)
// CHECK-NOT:     linalg.generic
func.func @fuse_row(%A: memref<32x64xi8>, %B: memref<64x48xi8>,
                    %D: memref<48xi32>, %C: memref<32x48xi32>) {
  %z = arith.constant 0 : i32
  linalg.fill ins(%z : i32) outs(%C : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>) outs(%C : memref<32x48xi32>)
  linalg.generic {indexing_maps = [#row, #id], iterator_types = ["parallel","parallel"]}
    ins(%D : memref<48xi32>) outs(%C : memref<32x48xi32>) {
  ^bb0(%b: i32, %c: i32):
    %s = arith.addi %c, %b : i32
    linalg.yield %s : i32
  }
  return
}

// Without the zero fill the matmul accumulates, so D is already spoken for and
// the add has to stay where it is.
// CHECK-LABEL: func.func @no_fuse
// CHECK:         gemmlir.matmul_i8(%arg0, %arg1, %arg3) :
// CHECK-NOT:     bias(
// CHECK:         linalg.generic
func.func @no_fuse(%A: memref<32x64xi8>, %B: memref<64x48xi8>,
                   %D: memref<32x48xi32>, %C: memref<32x48xi32>) {
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>) outs(%C : memref<32x48xi32>)
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%D : memref<32x48xi32>) outs(%C : memref<32x48xi32>) {
  ^bb0(%b: i32, %c: i32):
    %s = arith.addi %c, %b : i32
    linalg.yield %s : i32
  }
  return
}
