// `torch.cat` on the channel axis is what an Inception block, a DenseNet layer
// and a detection neck are joined with. It bufferizes into an allocation per
// branch plus a strided copy into the join, and `tiled_conv_stride_auto` takes
// the distance between two output pixels -- so the convolution writes the join
// itself and the copy disappears. Two 2048-element strided i8 copies cost
// 4.61 ms of a 4.96 ms inference on the board, because a strided `memref.copy`
// goes through the runtime's element-at-a-time walk.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s

// CHECK-LABEL: func.func @writes_into_a_join
// CHECK:         %[[S:.*]] = memref.subview %{{.*}}[0, 0, 0, 8] [1, 8, 8, 8] [1, 1, 1, 1]
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %[[S]])
// CHECK-NOT:     memref.copy
func.func @writes_into_a_join(%in: memref<1x8x8x8xi8>, %w: memref<1x1x8x8xi8>,
                              %join: memref<1x8x8x16xi8>) {
  %buf = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%in, %w, %buf) : (memref<1x8x8x8xi8>, memref<1x1x8x8xi8>, memref<1x8x8x8xi8>)
  %slice = memref.subview %join[0, 0, 0, 8] [1, 8, 8, 8] [1, 1, 1, 1]
           : memref<1x8x8x16xi8> to memref<1x8x8x8xi8, strided<[1024, 128, 16, 1], offset: 8>>
  memref.copy %buf, %slice : memref<1x8x8x8xi8> to memref<1x8x8x8xi8, strided<[1024, 128, 16, 1], offset: 8>>
  memref.dealloc %buf : memref<1x8x8x8xi8>
  return
}

// A window that is not a whole-channel slice is not a join, and the runtime has
// one number for the pixel stride: a spatial window would need two.
// CHECK-LABEL: func.func @not_a_join
// CHECK:         gemmlir.conv2d_i8
// CHECK:         memref.copy
func.func @not_a_join(%in: memref<1x8x8x8xi8>, %w: memref<1x1x8x8xi8>,
                      %join: memref<1x16x16x8xi8>) {
  %buf = memref.alloc() : memref<1x8x8x8xi8>
  gemmlir.conv2d_i8(%in, %w, %buf) : (memref<1x8x8x8xi8>, memref<1x1x8x8xi8>, memref<1x8x8x8xi8>)
  %slice = memref.subview %join[0, 4, 4, 0] [1, 8, 8, 8] [1, 1, 1, 1]
           : memref<1x16x16x8xi8> to memref<1x8x8x8xi8, strided<[2048, 128, 8, 1], offset: 544>>
  memref.copy %buf, %slice : memref<1x8x8x8xi8> to memref<1x8x8x8xi8, strided<[2048, 128, 8, 1], offset: 544>>
  memref.dealloc %buf : memref<1x8x8x8xi8>
  return
}

// A branch of a join is not always a convolution: ShuffleNet's unit passes half
// the channels through, so the piece joined in is whatever requantized them. An
// operation that writes every element of its destination can be pointed at the
// join too.
// CHECK-LABEL: func.func @elementwise_writes_a_join
// CHECK:         %[[S:.*]] = memref.subview %arg1[0, 0, 0, 0] [1, 8, 8, 8] [1, 1, 1, 1]
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%[[S]]
// CHECK-NOT:     memref.copy
func.func @elementwise_writes_a_join(%in: memref<1x8x8x8xi8>, %join: memref<1x8x8x16xi8>) {
  %s = arith.constant 2.000000e-02 : f32
  %t = arith.constant 4.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %buf = memref.alloc() : memref<1x8x8x8xi8>
  linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                                   affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
                  iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%in : memref<1x8x8x8xi8>) outs(%buf : memref<1x8x8x8xi8>) {
  ^bb0(%x: i8, %o: i8):
    %w = arith.sitofp %x : i8 to f32
    %m = arith.mulf %w, %s : f32
    %d = arith.divf %m, %t : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %v = arith.trunci %ch : i32 to i8
    linalg.yield %v : i8
  }
  %slice = memref.subview %join[0, 0, 0, 0] [1, 8, 8, 8] [1, 1, 1, 1]
           : memref<1x8x8x16xi8> to memref<1x8x8x8xi8, strided<[1024, 128, 16, 1]>>
  memref.copy %buf, %slice : memref<1x8x8x8xi8> to memref<1x8x8x8xi8, strided<[1024, 128, 16, 1]>>
  memref.dealloc %buf : memref<1x8x8x8xi8>
  return
}

// Pointing a convolution at one slice of a wider buffer moves its write earlier,
// to where the convolution is. Anything between the two that also touches the
// buffer then acts on the slice at the wrong time -- and a zero fill of the
// join sitting there erases what the convolution just wrote. It is a silent
// wrong answer, not a crash: measured, a ResNeXt block came back at 0.1029
// relative L2 where the same block ungrouped is 0.0026, because the branch
// reading that buffer was reading zeros.
//
// A fill of the whole buffer is the one case that can be kept, by running it
// before the convolution instead: the slice it writes was going to be
// overwritten either way.
// CHECK-LABEL: func @fill_of_the_join_runs_first
// CHECK:         linalg.fill
// CHECK-SAME:      outs(%[[J:.*]] : memref<1x16x16x64xi8>)
// CHECK:         %[[W:.*]] = memref.subview %[[J]][0, 0, 0, 0]
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %[[W]])
// The copy into the slice is gone; the one that follows is the join leaving.
// CHECK-NOT:     memref.copy {{.*}} to memref<1x16x16x32xi8
func.func @fill_of_the_join_runs_first(%in: memref<1x16x16x8xi8>, %f: memref<1x1x8x32xi8>,
                                       %out: memref<1x16x16x64xi8>) {
  %z = arith.constant 0 : i8
  %mid = memref.alloc() : memref<1x16x16x32xi8>
  gemmlir.conv2d_i8(%in, %f, %mid) {scale = 2.000000e-02 : f32}
    : (memref<1x16x16x8xi8>, memref<1x1x8x32xi8>, memref<1x16x16x32xi8>)
  %join = memref.alloc() : memref<1x16x16x64xi8>
  linalg.fill ins(%z : i8) outs(%join : memref<1x16x16x64xi8>)
  %w = memref.subview %join[0, 0, 0, 0] [1, 16, 16, 32] [1, 1, 1, 1]
     : memref<1x16x16x64xi8> to memref<1x16x16x32xi8, strided<[16384, 1024, 64, 1]>>
  memref.copy %mid, %w : memref<1x16x16x32xi8> to memref<1x16x16x32xi8, strided<[16384, 1024, 64, 1]>>
  memref.copy %join, %out : memref<1x16x16x64xi8> to memref<1x16x16x64xi8>
  memref.dealloc %mid : memref<1x16x16x32xi8>
  memref.dealloc %join : memref<1x16x16x64xi8>
  return
}

// Anything else on that buffer in between stays a copy: the convolution's write
// cannot be moved across an operation whose order matters.
// CHECK-LABEL: func @other_writer_stays_a_copy
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %alloc)
// CHECK:         memref.copy
func.func @other_writer_stays_a_copy(%in: memref<1x16x16x8xi8>, %f: memref<1x1x8x32xi8>,
                                     %other: memref<1x16x16x64xi8>) {
  %mid = memref.alloc() : memref<1x16x16x32xi8>
  gemmlir.conv2d_i8(%in, %f, %mid) {scale = 2.000000e-02 : f32}
    : (memref<1x16x16x8xi8>, memref<1x1x8x32xi8>, memref<1x16x16x32xi8>)
  %join = memref.alloc() : memref<1x16x16x64xi8>
  memref.copy %other, %join : memref<1x16x16x64xi8> to memref<1x16x16x64xi8>
  %w = memref.subview %join[0, 0, 0, 0] [1, 16, 16, 32] [1, 1, 1, 1]
     : memref<1x16x16x64xi8> to memref<1x16x16x32xi8, strided<[16384, 1024, 64, 1]>>
  memref.copy %mid, %w : memref<1x16x16x32xi8> to memref<1x16x16x32xi8, strided<[16384, 1024, 64, 1]>>
  memref.copy %join, %other : memref<1x16x16x64xi8> to memref<1x16x16x64xi8>
  memref.dealloc %mid : memref<1x16x16x32xi8>
  memref.dealloc %join : memref<1x16x16x64xi8>
  return
}

// A copy between two allocations where the source is written once and read
// nowhere else: the writer can write the target instead.
// `--materialize-pad-sources` puts one of these in on purpose, to keep a
// convolution's result out of the padded buffer it feeds, and once the padding
// has been folded away there is nothing left for it to separate.
// CHECK-LABEL: func @redundant_copy_between_allocations
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %[[D:[a-z_0-9]+]])
// CHECK-NOT:     memref.copy
// CHECK:         gemmlir.conv2d_i8(%[[D]], %arg2
func.func @redundant_copy_between_allocations(%in: memref<1x16x16x8xi8>,
    %f1: memref<1x1x8x32xi8>, %f2: memref<1x1x32x8xi8>, %out: memref<1x16x16x8xi8>) {
  %a = memref.alloc() : memref<1x16x16x32xi8>
  gemmlir.conv2d_i8(%in, %f1, %a) {scale = 2.000000e-02 : f32}
    : (memref<1x16x16x8xi8>, memref<1x1x8x32xi8>, memref<1x16x16x32xi8>)
  %b = memref.alloc() : memref<1x16x16x32xi8>
  memref.copy %a, %b : memref<1x16x16x32xi8> to memref<1x16x16x32xi8>
  gemmlir.conv2d_i8(%b, %f2, %out) {scale = 2.000000e-02 : f32}
    : (memref<1x16x16x32xi8>, memref<1x1x32x8xi8>, memref<1x16x16x8xi8>)
  memref.dealloc %a : memref<1x16x16x32xi8>
  memref.dealloc %b : memref<1x16x16x32xi8>
  return
}

// ... but only when everything that fill reads is already available up there,
// and a linalg op does not read only through its operands. `linalg.map`
// filling a buffer with a scalar **captures** that scalar in its region, where
// `getOperands` cannot see it. Hoisting such a fill above the value it yields
// produced IR that did not verify -- found by torchvision's SqueezeNet, whose
// ceil-mode max-pool pads with a computed one.
// CHECK-LABEL: func @fill_that_captures_a_later_value
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %[[M:[a-z0-9_]+]])
// CHECK:         arith.trunci
// CHECK:         linalg.map
// The convolution keeps its own buffer and the copy stays.
// CHECK:         memref.copy %[[M]]
func.func @fill_that_captures_a_later_value(%in: memref<1x16x16x8xi8>, %f: memref<1x1x8x32xi8>,
                                            %n: i32, %out: memref<1x16x16x64xi8>) {
  %mid = memref.alloc() : memref<1x16x16x32xi8>
  gemmlir.conv2d_i8(%in, %f, %mid) {scale = 2.000000e-02 : f32}
    : (memref<1x16x16x8xi8>, memref<1x1x8x32xi8>, memref<1x16x16x32xi8>)
  %lo = arith.constant -128 : i32
  %c = arith.maxsi %n, %lo : i32
  %v = arith.trunci %c : i32 to i8
  %join = memref.alloc() : memref<1x16x16x64xi8>
  linalg.map outs(%join : memref<1x16x16x64xi8>)
    (%init: i8) {
      linalg.yield %v : i8
    }
  %w = memref.subview %join[0, 0, 0, 0] [1, 16, 16, 32] [1, 1, 1, 1]
     : memref<1x16x16x64xi8> to memref<1x16x16x32xi8, strided<[16384, 1024, 64, 1]>>
  memref.copy %mid, %w : memref<1x16x16x32xi8> to memref<1x16x16x32xi8, strided<[16384, 1024, 64, 1]>>
  memref.copy %join, %out : memref<1x16x16x64xi8> to memref<1x16x16x64xi8>
  memref.dealloc %mid : memref<1x16x16x32xi8>
  memref.dealloc %join : memref<1x16x16x64xi8>
  return
}

// The same fill, with the scalar it captures defined before the convolution:
// now there is nothing in the way and the convolution writes its slice.
// CHECK-LABEL: func @fill_that_captures_an_earlier_value
// CHECK:         linalg.map
// CHECK:         %[[W:.*]] = memref.subview
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %[[W]])
// CHECK-NOT:     memref.copy {{.*}} to memref<1x16x16x32xi8
func.func @fill_that_captures_an_earlier_value(%in: memref<1x16x16x8xi8>, %f: memref<1x1x8x32xi8>,
                                               %n: i32, %out: memref<1x16x16x64xi8>) {
  %lo = arith.constant -128 : i32
  %c = arith.maxsi %n, %lo : i32
  %v = arith.trunci %c : i32 to i8
  %mid = memref.alloc() : memref<1x16x16x32xi8>
  gemmlir.conv2d_i8(%in, %f, %mid) {scale = 2.000000e-02 : f32}
    : (memref<1x16x16x8xi8>, memref<1x1x8x32xi8>, memref<1x16x16x32xi8>)
  %join = memref.alloc() : memref<1x16x16x64xi8>
  linalg.map outs(%join : memref<1x16x16x64xi8>)
    (%init: i8) {
      linalg.yield %v : i8
    }
  %w = memref.subview %join[0, 0, 0, 0] [1, 16, 16, 32] [1, 1, 1, 1]
     : memref<1x16x16x64xi8> to memref<1x16x16x32xi8, strided<[16384, 1024, 64, 1]>>
  memref.copy %mid, %w : memref<1x16x16x32xi8> to memref<1x16x16x32xi8, strided<[16384, 1024, 64, 1]>>
  memref.copy %join, %out : memref<1x16x16x64xi8> to memref<1x16x16x64xi8>
  memref.dealloc %mid : memref<1x16x16x32xi8>
  memref.dealloc %join : memref<1x16x16x64xi8>
  return
}
