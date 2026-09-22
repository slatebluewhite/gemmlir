// PyTorch's `ceil_mode` asks for an output one step wider than the input
// supports, and torch-mlir pays for it with a `tensor.pad` of `stride - 1` on
// the high side. Where the window divides the input evenly the ceiling and the
// floor agree and not one padded element is read.

// RUN: gemmlir-opt --drop-unread-padding %s | FileCheck %s

// SqueezeNet's first pool: 31x31 padded to 33x33, window 3, stride 2, output
// 15x15. The highest index read is (15-1)*2 + (3-1) = 30, and the source has
// 0..30 -- so the two padded rows and columns are never reached. What they cost
// is a zero-fill of 69696 bytes, a copy of 61504, and the pool's chance to ride
// out on the convolution's own `mvout`, because the copy stands between them.
// CHECK-LABEL: func.func @ceil_mode_padding_is_never_read
// CHECK-NOT:     tensor.pad
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      ins(%arg0, %{{.*}} : tensor<1x31x31x64xi8>, tensor<3x3xf32>)
func.func @ceil_mode_padding_is_never_read(%in: tensor<1x31x31x64xi8>)
    -> tensor<1x15x15x64xi8> {
  %z = arith.constant 0 : i8
  %least = arith.constant -128 : i8
  %padded = tensor.pad %in low[0, 0, 0, 0] high[0, 2, 2, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %z : i8
  } : tensor<1x31x31x64xi8> to tensor<1x33x33x64xi8>
  %w = tensor.empty() : tensor<3x3xf32>
  %e = tensor.empty() : tensor<1x15x15x64xi8>
  %f = linalg.fill ins(%least : i8) outs(%e : tensor<1x15x15x64xi8>) -> tensor<1x15x15x64xi8>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%padded, %w : tensor<1x33x33x64xi8>, tensor<3x3xf32>)
    outs(%f : tensor<1x15x15x64xi8>) -> tensor<1x15x15x64xi8>
  return %p : tensor<1x15x15x64xi8>
}

// -----

// One row further and the pad *is* read: output 16 reaches index
// (16-1)*2 + 2 = 32, which is outside a 31-wide source. The pad stays.
// CHECK-LABEL: func.func @padding_that_is_read_stays
// CHECK:         tensor.pad
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      tensor<1x33x33x64xi8>
func.func @padding_that_is_read_stays(%in: tensor<1x31x31x64xi8>)
    -> tensor<1x16x16x64xi8> {
  %z = arith.constant 0 : i8
  %least = arith.constant -128 : i8
  %padded = tensor.pad %in low[0, 0, 0, 0] high[0, 2, 2, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %z : i8
  } : tensor<1x31x31x64xi8> to tensor<1x33x33x64xi8>
  %w = tensor.empty() : tensor<3x3xf32>
  %e = tensor.empty() : tensor<1x16x16x64xi8>
  %f = linalg.fill ins(%least : i8) outs(%e : tensor<1x16x16x64xi8>) -> tensor<1x16x16x64xi8>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%padded, %w : tensor<1x33x33x64xi8>, tensor<3x3xf32>)
    outs(%f : tensor<1x16x16x64xi8>) -> tensor<1x16x16x64xi8>
  return %p : tensor<1x16x16x64xi8>
}

// -----

// A border on the **low** side shifts every index the pool reads, so the
// arithmetic above does not apply to it -- that is a convolution's padding, and
// it is folded into the call elsewhere. Left alone.
// CHECK-LABEL: func.func @a_low_border_is_left_alone
// CHECK:         tensor.pad
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      tensor<1x33x33x64xi8>
func.func @a_low_border_is_left_alone(%in: tensor<1x31x31x64xi8>)
    -> tensor<1x15x15x64xi8> {
  %z = arith.constant 0 : i8
  %least = arith.constant -128 : i8
  %padded = tensor.pad %in low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %z : i8
  } : tensor<1x31x31x64xi8> to tensor<1x33x33x64xi8>
  %w = tensor.empty() : tensor<3x3xf32>
  %e = tensor.empty() : tensor<1x15x15x64xi8>
  %f = linalg.fill ins(%least : i8) outs(%e : tensor<1x15x15x64xi8>) -> tensor<1x15x15x64xi8>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%padded, %w : tensor<1x33x33x64xi8>, tensor<3x3xf32>)
    outs(%f : tensor<1x15x15x64xi8>) -> tensor<1x15x15x64xi8>
  return %p : tensor<1x15x15x64xi8>
}

// -----

// The same arithmetic on an NCHW pool: 15x15 padded to 17x17, output 7x7 reads
// up to (7-1)*2 + 2 = 14, inside a 15-wide source.
// CHECK-LABEL: func.func @nchw_ceil_mode_padding
// CHECK-NOT:     tensor.pad
// CHECK:         linalg.pooling_nchw_max
// CHECK-SAME:      ins(%arg0, %{{.*}} : tensor<1x128x15x15xi8>, tensor<3x3xf32>)
func.func @nchw_ceil_mode_padding(%in: tensor<1x128x15x15xi8>)
    -> tensor<1x128x7x7xi8> {
  %z = arith.constant 0 : i8
  %least = arith.constant -128 : i8
  %padded = tensor.pad %in low[0, 0, 0, 0] high[0, 0, 2, 2] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %z : i8
  } : tensor<1x128x15x15xi8> to tensor<1x128x17x17xi8>
  %w = tensor.empty() : tensor<3x3xf32>
  %e = tensor.empty() : tensor<1x128x7x7xi8>
  %f = linalg.fill ins(%least : i8) outs(%e : tensor<1x128x7x7xi8>) -> tensor<1x128x7x7xi8>
  %p = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%padded, %w : tensor<1x128x17x17xi8>, tensor<3x3xf32>)
    outs(%f : tensor<1x128x7x7xi8>) -> tensor<1x128x7x7xi8>
  return %p : tensor<1x128x7x7xi8>
}
