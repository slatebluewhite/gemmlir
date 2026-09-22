// RUN: gemmlir-opt --pack-int8-table-lookup --split-input-file %s | FileCheck %s

// An i8 -> i8 table sweep is 72% of EfficientNet's elementwise work and 29% of
// a ViT's. `--table-for-i8-elementwise` leaves it as four instructions an
// element -- `lb`, `add`, `lbu`, `sb` -- and it measures about 11 cycles, so
// something is waiting. The chain is two *dependent* loads: the byte, and then
// the table entry that byte indexes.
//
// One 8-byte load takes the first of them out of the chain. Measured on the
// board at EfficientNet's shape, 16x16x576: **-44.5%**. Packing the eight
// stores back into a word as well is worse (-37.7%), so the outputs stay bytes.

#id = affine_map<(d0, d1) -> (d0, d1)>

memref.global "private" constant @tbl : memref<256xi8> = dense<7>

// `sext(q) + 128` is `u ^ 0x80` on the unsigned byte in the word, so one `xor`
// of the whole word does all eight indices.
// CHECK-LABEL: func.func @a_a_table_reads_a_word
// CHECK-DAG:   %[[HI:.*]] = arith.constant -9187201950435737472 : i64
// CHECK:       %[[W:.*]] = memref.view {{.*}} to memref<8xi64>
// CHECK:       scf.for
// CHECK:         %[[V:.*]] = memref.load %[[W]]
// CHECK:         %[[X:.*]] = arith.xori %[[V]], %[[HI]]
// Eight lookups from one load, and no `arith.extsi` of a byte left:
// CHECK-COUNT-8: memref.load %{{.*}}[%{{.*}}] : memref<256xi8>
// CHECK-NOT:   arith.extsi
// CHECK-NOT:   linalg.generic
func.func @a_a_table_reads_a_word(%src: memref<4x16xi8>, %dst: memref<4x16xi8>) {
  %c128 = arith.constant 128 : i32
  %t = memref.get_global @tbl : memref<256xi8>
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%src : memref<4x16xi8>) outs(%dst : memref<4x16xi8>) {
  ^bb0(%in: i8, %o: i8):
    %e = arith.extsi %in : i8 to i32
    %b = arith.addi %e, %c128 : i32
    %i = arith.index_cast %b : i32 to index
    %v = memref.load %t[%i] : memref<256xi8>
    linalg.yield %v : i8
  }
  return
}

// -----

// Bufferizing a 4-D map on a **batch of one** turns `d0` into a constant `0`,
// so `isIdentity()` says no to an ordinary elementwise loop. Walking the map is
// what makes this fire on EfficientNet at all -- it is written that way on all
// 49 of its tables.

#batch1 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#id4    = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

memref.global "private" constant @tbl_b : memref<256xi8> = dense<3>

// CHECK-LABEL: func.func @b_a_batch_of_one
// CHECK:       memref.view {{.*}} to memref<4xi64>
// CHECK-NOT:   linalg.generic
func.func @b_a_batch_of_one(%src: memref<1x2x2x8xi8>, %dst: memref<1x2x2x8xi8>) {
  %c128 = arith.constant 128 : i32
  %t = memref.get_global @tbl_b : memref<256xi8>
  linalg.generic {indexing_maps = [#batch1, #id4],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%src : memref<1x2x2x8xi8>) outs(%dst : memref<1x2x2x8xi8>) {
  ^bb0(%in: i8, %o: i8):
    %e = arith.extsi %in : i8 to i32
    %b = arith.addi %e, %c128 : i32
    %i = arith.index_cast %b : i32 to index
    %v = memref.load %t[%i] : memref<256xi8>
    linalg.yield %v : i8
  }
  return
}

// -----

// A count that is not a whole number of words has no word to read. EfficientNet's
// squeeze branch is 20 and 28 channels wide, and those keep the byte loop.

#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

memref.global "private" constant @tbl_c : memref<256xi8> = dense<1>

// CHECK-LABEL: func.func @c_not_a_whole_word
// CHECK:       linalg.generic
// CHECK:       arith.extsi
func.func @c_not_a_whole_word(%src: memref<1x1x1x20xi8>, %dst: memref<1x1x1x20xi8>) {
  %c128 = arith.constant 128 : i32
  %t = memref.get_global @tbl_c : memref<256xi8>
  linalg.generic {indexing_maps = [#id4, #id4],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%src : memref<1x1x1x20xi8>) outs(%dst : memref<1x1x1x20xi8>) {
  ^bb0(%in: i8, %o: i8):
    %e = arith.extsi %in : i8 to i32
    %b = arith.addi %e, %c128 : i32
    %i = arith.index_cast %b : i32 to index
    %v = memref.load %t[%i] : memref<256xi8>
    linalg.yield %v : i8
  }
  return
}

// -----

// A strided buffer does not start where it says, and `memref.view` takes an
// identity `memref<?xi8>`.

#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

memref.global "private" constant @tbl_d : memref<256xi8> = dense<1>

// CHECK-LABEL: func.func @d_a_strided_slice
// CHECK:       linalg.generic
// CHECK:       arith.extsi
func.func @d_a_strided_slice(%src: memref<1x2x2x8xi8>, %big: memref<1x2x2x16xi8>) {
  %c128 = arith.constant 128 : i32
  %t = memref.get_global @tbl_d : memref<256xi8>
  %dst = memref.subview %big[0, 0, 0, 0] [1, 2, 2, 8] [1, 1, 1, 1]
      : memref<1x2x2x16xi8> to memref<1x2x2x8xi8, strided<[64, 32, 16, 1]>>
  linalg.generic {indexing_maps = [#id4, #id4],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%src : memref<1x2x2x8xi8>) outs(%dst : memref<1x2x2x8xi8, strided<[64, 32, 16, 1]>>) {
  ^bb0(%in: i8, %o: i8):
    %e = arith.extsi %in : i8 to i32
    %b = arith.addi %e, %c128 : i32
    %i = arith.index_cast %b : i32 to index
    %v = memref.load %t[%i] : memref<256xi8>
    linalg.yield %v : i8
  }
  return
}

// -----

// The body has to be the lookup, not something that merely contains one: a
// table indexed by a *computed* value is a gather this pass has no word for.

#id = affine_map<(d0) -> (d0)>

memref.global "private" constant @tbl_e : memref<256xi8> = dense<1>

// CHECK-LABEL: func.func @e_not_the_shape_of_a_table
// CHECK:       linalg.generic
// CHECK:       arith.muli
func.func @e_not_the_shape_of_a_table(%src: memref<16xi8>, %dst: memref<16xi8>) {
  %c128 = arith.constant 128 : i32
  %c2 = arith.constant 2 : i32
  %t = memref.get_global @tbl_e : memref<256xi8>
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%src : memref<16xi8>) outs(%dst : memref<16xi8>) {
  ^bb0(%in: i8, %o: i8):
    %e = arith.extsi %in : i8 to i32
    %m = arith.muli %e, %c2 : i32
    %b = arith.addi %m, %c128 : i32
    %i = arith.index_cast %b : i32 to index
    %v = memref.load %t[%i] : memref<256xi8>
    linalg.yield %v : i8
  }
  return
}
