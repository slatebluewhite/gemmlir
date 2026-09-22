// Two things want the temporaries static.
//
// Gemmini returns wrong data on the U280 board when the buffer it writes moves
// between calls -- the CNN was stable until folding a pooling away removed one
// allocation and made another alternate between two bins. And every call that
// allocates and frees is a fresh mapping the kernel has to populate, which on a
// MobileNetV2 whose temporaries come to 4.45 MB was 56 ms an inference.

// RUN: gemmlir-opt --plan-static-buffers %s | FileCheck %s

// One arena per function that needs one. Each buffer gets its own space:
// letting buffers whose lives do not overlap share it is not safe on this
// board -- see the pass description and docs/pipeline.md.
// CHECK-DAG: memref.global "private" @[[A:.*]]accelerator_buffer{{.*}} : memref<8192xi8> = uninitialized
// CHECK-DAG: memref.global "private" @{{.*}}three_temporaries{{.*}} : memref<3072xi8> = uninitialized

// CHECK-LABEL: func.func @accelerator_buffer
// CHECK:         %[[G:.*]] = memref.get_global @[[A]]
// CHECK:         %[[V:.*]] = memref.view %[[G]]
// CHECK-SAME:      to memref<8x256xi32>
// CHECK:         gemmlir.matmul_i8(%arg0, %arg1, %[[V]])
// CHECK-NOT:     memref.dealloc %[[V]]
func.func @accelerator_buffer(%a: memref<8x27xi8>, %b: memref<27x256xi8>) -> memref<8x256xi32> {
  %acc = memref.alloc() {alignment = 64 : i64} : memref<8x256xi32>
  gemmlir.matmul_i8(%a, %b, %acc) : (memref<8x27xi8> x memref<27x256xi8>) -> memref<8x256xi32> {accumulate = false}
  %out = memref.alloc() : memref<8x256xi32>
  memref.copy %acc, %out : memref<8x256xi32> to memref<8x256xi32>
  memref.dealloc %acc : memref<8x256xi32>
  return %out : memref<8x256xi32>
}

// Ownership of a returned buffer is the caller's, so it keeps the allocator's
// lifetime even though the accelerator wrote it.
// CHECK-LABEL: func.func @returned_buffer
// CHECK:         memref.alloc()
// CHECK:         gemmlir.matmul_i8
// CHECK:         return
func.func @returned_buffer(%a: memref<8x27xi8>, %b: memref<27x256xi8>) -> memref<8x256xi32> {
  %acc = memref.alloc() {alignment = 64 : i64} : memref<8x256xi32>
  gemmlir.matmul_i8(%a, %b, %acc) : (memref<8x27xi8> x memref<27x256xi8>) -> memref<8x256xi32> {accumulate = false}
  return %acc : memref<8x256xi32>
}

// Three 1 KiB temporaries take 3 KiB, at their own offsets, and the deallocs
// go: the arena is never handed back, so neither are its pieces.
// CHECK-LABEL: func.func @three_temporaries
// CHECK:         memref.view %{{.*}}[%c0]
// CHECK:         memref.view %{{.*}}[%c1024]
// CHECK:         memref.view %{{.*}}[%c2048]
// CHECK-NOT:     memref.dealloc
func.func @three_temporaries(%v: i8) {
  %a = memref.alloc() : memref<1024xi8>
  linalg.fill ins(%v : i8) outs(%a : memref<1024xi8>)
  memref.dealloc %a : memref<1024xi8>
  %b = memref.alloc() : memref<1024xi8>
  linalg.fill ins(%v : i8) outs(%b : memref<1024xi8>)
  memref.dealloc %b : memref<1024xi8>
  %c = memref.alloc() : memref<1024xi8>
  linalg.fill ins(%v : i8) outs(%c : memref<1024xi8>)
  memref.dealloc %c : memref<1024xi8>
  return
}
