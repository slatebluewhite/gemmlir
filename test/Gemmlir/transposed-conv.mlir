// torch-mlir writes a transposed convolution as an ordinary one over an input
// with `stride - 1` zeros inserted between its samples: a zero-filled buffer and
// a strided `tensor.insert_slice` into it. The accelerator does that with
// `input_dilation`, so the buffer never has to exist.

// RUN: gemmlir-opt --hoist-elementwise-before-gather %s | FileCheck %s
// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s --check-prefix=FOLD

#nhwc = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Quantizing after the stuffing converts the whole stuffed buffer -- four times
// the elements for a stride of two, three quarters of them zeros. Quantizing
// first converts only the real ones, and the stuffing then moves i8. Sound
// because the quantization takes zero to zero.
// CHECK-LABEL: func.func @hoist_over_stuffing
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<1x8x8x4xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<1x8x8x4xi8>)
// CHECK:         %[[Z:.*]] = linalg.fill ins(%{{.*}} : i8)
// CHECK:         tensor.insert_slice %[[Q]] into %[[Z]][0, 1, 1, 0] [1, 8, 8, 4] [1, 2, 2, 1]
func.func @hoist_over_stuffing(%in: tensor<1x8x8x4xf32>) -> tensor<1x17x17x4xi8> {
  %zero = arith.constant 0.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<1x17x17x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<1x17x17x4xf32>) -> tensor<1x17x17x4xf32>
  %stuffed = tensor.insert_slice %in into %f[0, 1, 1, 0] [1, 8, 8, 4] [1, 2, 2, 1]
             : tensor<1x8x8x4xf32> into tensor<1x17x17x4xf32>
  %o = tensor.empty() : tensor<1x17x17x4xi8>
  %q = linalg.generic {indexing_maps = [#nhwc, #nhwc],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%stuffed : tensor<1x17x17x4xf32>) outs(%o : tensor<1x17x17x4xi8>) {
  ^bb0(%x: f32, %out: i8):
    %d = arith.divf %x, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x17x17x4xi8>
  return %q : tensor<1x17x17x4xi8>
}

// And once it is in i8 next to the call, the stuffing is the call's own
// `input_dilation`: the 17x17 buffer disappears and the convolution reads the
// 8x8 input. The border it was inserted at is the padding.
// FOLD-LABEL: func.func @fold_stuffing
// FOLD:         gemmlir.conv2d_i8(%arg0, %arg1, %arg2)
// FOLD-SAME:      input_dilation = 2
// FOLD-SAME:      padding = 1
// FOLD-NOT:     memref.subview
func.func @fold_stuffing(%in: memref<1x8x8x4xi8>, %w: memref<2x2x4x4xi8>,
                         %out: memref<1x16x16x4xi8>) {
  %z = arith.constant 0 : i8
  %buf = memref.alloc() : memref<1x17x17x4xi8>
  linalg.fill ins(%z : i8) outs(%buf : memref<1x17x17x4xi8>)
  %win = memref.subview %buf[0, 1, 1, 0] [1, 8, 8, 4] [1, 2, 2, 1]
         : memref<1x17x17x4xi8> to memref<1x8x8x4xi8, strided<[1156, 136, 8, 1], offset: 72>>
  memref.copy %in, %win : memref<1x8x8x4xi8> to memref<1x8x8x4xi8, strided<[1156, 136, 8, 1], offset: 72>>
  gemmlir.conv2d_i8(%buf, %w, %out) : (memref<1x17x17x4xi8>, memref<2x2x4x4xi8>, memref<1x16x16x4xi8>)
  memref.dealloc %buf : memref<1x17x17x4xi8>
  return
}

// The runtime's accelerator path takes an input dilation of 2 and only with a
// unit stride; a stuffed input under a strided convolution stays a buffer.
// FOLD-LABEL: func.func @stuffing_under_a_stride
// FOLD:         memref.subview
// FOLD:         gemmlir.conv2d_i8
// FOLD-NOT:     input_dilation
func.func @stuffing_under_a_stride(%in: memref<1x8x8x4xi8>, %w: memref<2x2x4x4xi8>,
                                   %out: memref<1x8x8x4xi8>) {
  %z = arith.constant 0 : i8
  %buf = memref.alloc() : memref<1x17x17x4xi8>
  linalg.fill ins(%z : i8) outs(%buf : memref<1x17x17x4xi8>)
  %win = memref.subview %buf[0, 1, 1, 0] [1, 8, 8, 4] [1, 2, 2, 1]
         : memref<1x17x17x4xi8> to memref<1x8x8x4xi8, strided<[1156, 136, 8, 1], offset: 72>>
  memref.copy %in, %win : memref<1x8x8x4xi8> to memref<1x8x8x4xi8, strided<[1156, 136, 8, 1], offset: 72>>
  gemmlir.conv2d_i8(%buf, %w, %out) {stride = 2 : i64} : (memref<1x17x17x4xi8>, memref<2x2x4x4xi8>, memref<1x8x8x4xi8>)
  memref.dealloc %buf : memref<1x17x17x4xi8>
  return
}

// The depthwise call has no input dilation at all, so its stuffing stays too.
// FOLD-LABEL: func.func @depthwise_keeps_its_buffer
// FOLD:         memref.subview
// FOLD:         gemmlir.depthwise_conv2d_i8
// FOLD-NOT:     input_dilation
func.func @depthwise_keeps_its_buffer(%in: memref<1x8x8x4xi8>, %w: memref<4x2x2xi8>,
                                      %out: memref<1x16x16x4xi8>) {
  %z = arith.constant 0 : i8
  %buf = memref.alloc() : memref<1x17x17x4xi8>
  linalg.fill ins(%z : i8) outs(%buf : memref<1x17x17x4xi8>)
  %win = memref.subview %buf[0, 1, 1, 0] [1, 8, 8, 4] [1, 2, 2, 1]
         : memref<1x17x17x4xi8> to memref<1x8x8x4xi8, strided<[1156, 136, 8, 1], offset: 72>>
  memref.copy %in, %win : memref<1x8x8x4xi8> to memref<1x8x8x4xi8, strided<[1156, 136, 8, 1], offset: 72>>
  gemmlir.depthwise_conv2d_i8(%buf, %w, %out) : (memref<1x17x17x4xi8>, memref<4x2x2xi8>, memref<1x16x16x4xi8>)
  memref.dealloc %buf : memref<1x17x17x4xi8>
  return
}
