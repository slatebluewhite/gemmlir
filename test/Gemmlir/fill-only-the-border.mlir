// RUN: gemmlir-opt %s --fill-only-the-border --split-input-file | FileCheck %s

// An NHWC padding: the whole buffer is filled and the real data is copied into
// the middle of it. Only the two slabs the copy misses need filling, and their
// union is the complement of the box exactly once -- the last two rows entire,
// then the last two columns of the rows above them.

// CHECK-LABEL: func.func @pad_high
// CHECK:         %[[B:.*]] = memref.alloc()
// CHECK:         %[[S0:.*]] = memref.subview %[[B]][0, 12, 0, 0] [1, 2, 14, 480] [1, 1, 1, 1]
// CHECK:         linalg.fill ins(%{{.*}} : f32) outs(%[[S0]]
// CHECK:         %[[S1:.*]] = memref.subview %[[B]][0, 0, 12, 0] [1, 12, 2, 480] [1, 1, 1, 1]
// CHECK:         linalg.fill ins(%{{.*}} : f32) outs(%[[S1]]
// CHECK:         memref.copy
func.func @pad_high(%src: memref<1x12x12x480xf32>) -> memref<1x14x14x480xf32> {
  %c = arith.constant 0xFF800000 : f32
  %b = memref.alloc() : memref<1x14x14x480xf32>
  linalg.fill ins(%c : f32) outs(%b : memref<1x14x14x480xf32>)
  %s = memref.subview %b[0, 0, 0, 0] [1, 12, 12, 480] [1, 1, 1, 1]
    : memref<1x14x14x480xf32> to memref<1x12x12x480xf32, strided<[94080, 6720, 480, 1]>>
  memref.copy %src, %s : memref<1x12x12x480xf32> to memref<1x12x12x480xf32, strided<[94080, 6720, 480, 1]>>
  return %b : memref<1x14x14x480xf32>
}

// -----

// A padding on both sides is four slabs, and the two for the inner dimension
// are clipped to the box on the outer one so no element is filled twice.

// CHECK-LABEL: func.func @pad_both
// CHECK-DAG:     memref.subview %{{.*}}[0, 0, 0, 0] [1, 1, 9, 512]
// CHECK-DAG:     memref.subview %{{.*}}[0, 7, 0, 0] [1, 2, 9, 512]
// CHECK-DAG:     memref.subview %{{.*}}[0, 1, 0, 0] [1, 6, 1, 512]
// CHECK-DAG:     memref.subview %{{.*}}[0, 1, 7, 0] [1, 6, 2, 512]
// CHECK-NOT:     linalg.fill ins(%{{.*}} : i8) outs(%alloc :
func.func @pad_both(%src: memref<1x6x6x512xi8>) -> memref<1x9x9x512xi8> {
  %c = arith.constant -128 : i8
  %b = memref.alloc() : memref<1x9x9x512xi8>
  linalg.fill ins(%c : i8) outs(%b : memref<1x9x9x512xi8>)
  %s = memref.subview %b[0, 1, 1, 0] [1, 6, 6, 512] [1, 1, 1, 1]
    : memref<1x9x9x512xi8> to memref<1x6x6x512xi8, strided<[41472, 4608, 512, 1], offset: 5120>>
  memref.copy %src, %s : memref<1x6x6x512xi8> to memref<1x6x6x512xi8, strided<[41472, 4608, 512, 1], offset: 5120>>
  return %b : memref<1x9x9x512xi8>
}

// -----

// The form a bufferized `tensor.pad` actually arrives in: a `linalg.map` with
// no inputs whose body yields the constant.

// CHECK-LABEL: func.func @pad_as_map
// CHECK-NOT:     linalg.map
// CHECK:         linalg.fill
func.func @pad_as_map(%src: memref<1x12x12x64xi8>) -> memref<1x14x14x64xi8> {
  %c = arith.constant -128 : i8
  %b = memref.alloc() : memref<1x14x14x64xi8>
  linalg.map outs(%b : memref<1x14x14x64xi8>)
    (%init: i8) {
      linalg.yield %c : i8
    }
  %s = memref.subview %b[0, 0, 0, 0] [1, 12, 12, 64] [1, 1, 1, 1]
    : memref<1x14x14x64xi8> to memref<1x12x12x64xi8, strided<[12544, 896, 64, 1]>>
  memref.copy %src, %s : memref<1x12x12x64xi8> to memref<1x12x12x64xi8, strided<[12544, 896, 64, 1]>>
  return %b : memref<1x14x14x64xi8>
}

// -----

// Something reads the buffer between the fill and the write, so the fill is not
// simply covered.

// CHECK-LABEL: func.func @read_in_between
// CHECK:         linalg.fill ins(%{{.*}} : f32) outs(%alloc :
func.func @read_in_between(%src: memref<1x12x12x480xf32>, %out: memref<1x14x14x480xf32>) {
  %c = arith.constant 0.0 : f32
  %b = memref.alloc() : memref<1x14x14x480xf32>
  linalg.fill ins(%c : f32) outs(%b : memref<1x14x14x480xf32>)
  memref.copy %b, %out : memref<1x14x14x480xf32> to memref<1x14x14x480xf32>
  %s = memref.subview %b[0, 0, 0, 0] [1, 12, 12, 480] [1, 1, 1, 1]
    : memref<1x14x14x480xf32> to memref<1x12x12x480xf32, strided<[94080, 6720, 480, 1]>>
  memref.copy %src, %s : memref<1x12x12x480xf32> to memref<1x12x12x480xf32, strided<[94080, 6720, 480, 1]>>
  memref.dealloc %b : memref<1x14x14x480xf32>
  return
}

// -----

// The write reads its own destination back, so the fill is its accumulator's
// starting value and has to stay whole.

// CHECK-LABEL: func.func @accumulated_into
// CHECK:         linalg.fill ins(%{{.*}} : f32) outs(%alloc :
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @accumulated_into(%src: memref<1x12x12x480xf32>) -> memref<1x14x14x480xf32> {
  %c = arith.constant 0.0 : f32
  %b = memref.alloc() : memref<1x14x14x480xf32>
  linalg.fill ins(%c : f32) outs(%b : memref<1x14x14x480xf32>)
  %s = memref.subview %b[0, 0, 0, 0] [1, 12, 12, 480] [1, 1, 1, 1]
    : memref<1x14x14x480xf32> to memref<1x12x12x480xf32, strided<[94080, 6720, 480, 1]>>
  linalg.generic {indexing_maps = [#map, #map],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%src : memref<1x12x12x480xf32>)
    outs(%s : memref<1x12x12x480xf32, strided<[94080, 6720, 480, 1]>>) {
  ^bb0(%in: f32, %acc: f32):
    %a = arith.addf %in, %acc : f32
    linalg.yield %a : f32
  }
  return %b : memref<1x14x14x480xf32>
}

// -----

// Too little to save: the slabs would cost more in loop nests than they spare.

// CHECK-LABEL: func.func @sliver
// CHECK:         linalg.fill ins(%{{.*}} : f32) outs(%alloc :
func.func @sliver(%src: memref<1x2x2x16xf32>) -> memref<1x8x8x16xf32> {
  %c = arith.constant 0.0 : f32
  %b = memref.alloc() : memref<1x8x8x16xf32>
  linalg.fill ins(%c : f32) outs(%b : memref<1x8x8x16xf32>)
  %s = memref.subview %b[0, 0, 0, 0] [1, 2, 2, 16] [1, 1, 1, 1]
    : memref<1x8x8x16xf32> to memref<1x2x2x16xf32, strided<[1024, 128, 16, 1]>>
  memref.copy %src, %s : memref<1x2x2x16xf32> to memref<1x2x2x16xf32, strided<[1024, 128, 16, 1]>>
  return %b : memref<1x8x8x16xf32>
}
