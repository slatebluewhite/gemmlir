// `tensor.pad` normally bufferizes into an allocation, a zero fill and a copy
// of the real input into the middle, and --convert-linalg-to-gemmlir turns
// exactly that shape into the runtime's own `padding`, so none of it survives.
//
// Once the padding is on i8 -- which is what --hoist-elementwise-before-gather
// makes of it, and much cheaper than padding f32 -- bufferization sees that it
// can write the producer straight into the middle of the padded buffer and
// skips the copy. That window is one the runtime cannot address: it takes one
// stride between pixels and derives the row from it. Both ends are then
// stranded, the producer because its destination is not addressable and the
// consumer because the shape the padding folds out of is not there.
//
// Whether bufferization takes that chance depends on the whole function, so
// this pins what the pass does rather than what bufferization then makes of it;
// the effect is in docs/pipeline.md, measured on the board.

// RUN: gemmlir-opt --materialize-pad-sources %s | FileCheck %s

// CHECK-LABEL: func @padded_producer
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK:         %[[OWN:.*]] = bufferization.alloc_tensor() copy(%[[Q]])
// CHECK:         tensor.pad %[[OWN]] low[0, 1, 1, 0] high[0, 1, 1, 0]
func.func @padded_producer(%acc: tensor<1x16x16x8xi32>, %f: tensor<3x3x8x8xi8>)
    -> tensor<1x16x16x8xi8> {
  %zero = arith.constant 0 : i8
  %z = arith.constant 0 : i32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x16x16x8xi8>
  // the requantization tail of whatever produced %acc
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%acc : tensor<1x16x16x8xi32>) outs(%e : tensor<1x16x16x8xi8>) {
  ^bb0(%a: i32, %b: i8):
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    %r = arith.fptosi %m : f32 to i32
    %c1 = arith.maxsi %r, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x16x16x8xi8>
  %padded = tensor.pad %q low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%i: index, %j: index, %k: index, %l: index):
    tensor.yield %zero : i8
  } : tensor<1x16x16x8xi8> to tensor<1x18x18x8xi8>
  %oe = tensor.empty() : tensor<1x16x16x8xi32>
  %oz = linalg.fill ins(%z : i32) outs(%oe : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%padded, %f : tensor<1x18x18x8xi8>, tensor<3x3x8x8xi8>)
       outs(%oz : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %re = tensor.empty() : tensor<1x16x16x8xi8>
  %out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%c : tensor<1x16x16x8xi32>) outs(%re : tensor<1x16x16x8xi8>) {
  ^bb0(%a: i32, %b: i8):
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    %r = arith.fptosi %m : f32 to i32
    %c1 = arith.maxsi %r, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x16x16x8xi8>
  return %out : tensor<1x16x16x8xi8>
}

// -----

// A padding no convolution reads is not in anyone's way.
// CHECK-LABEL: func @not_for_a_convolution
// CHECK-NOT:     bufferization.alloc_tensor
func.func @not_for_a_convolution(%acc: tensor<1x16x16x8xi32>) -> tensor<1x18x18x8xi8> {
  %zero = arith.constant 0 : i8
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<1x16x16x8xi8>
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%acc : tensor<1x16x16x8xi32>) outs(%e : tensor<1x16x16x8xi8>) {
  ^bb0(%a: i32, %b: i8):
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    %r = arith.fptosi %m : f32 to i8
    linalg.yield %r : i8
  } -> tensor<1x16x16x8xi8>
  %padded = tensor.pad %q low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%i: index, %j: index, %k: index, %l: index):
    tensor.yield %zero : i8
  } : tensor<1x16x16x8xi8> to tensor<1x18x18x8xi8>
  return %padded : tensor<1x18x18x8xi8>
}

// -----

// An f32 padding is not the case this is about: the accelerator never reads it.
// CHECK-LABEL: func @f32_padding
// CHECK-NOT:     bufferization.alloc_tensor
func.func @f32_padding(%in: tensor<1x16x16x8xf32>, %f: tensor<3x3x8x8xf32>)
    -> tensor<1x16x16x8xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x16x16x8xf32>
  %q = linalg.copy ins(%in : tensor<1x16x16x8xf32>) outs(%e : tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>
  %padded = tensor.pad %q low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%i: index, %j: index, %k: index, %l: index):
    tensor.yield %zero : f32
  } : tensor<1x16x16x8xf32> to tensor<1x18x18x8xf32>
  %oe = tensor.empty() : tensor<1x16x16x8xf32>
  %oz = linalg.fill ins(%zero : f32) outs(%oe : tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%padded, %f : tensor<1x18x18x8xf32>, tensor<3x3x8x8xf32>)
       outs(%oz : tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>
  return %c : tensor<1x16x16x8xf32>
}

// -----

// With another consumer bufferization has to keep the producer's own buffer
// anyway, so asking for one here would only add a copy -- nine of them on
// ResNet-20, whose every block hands its result to the shortcut as well.
// CHECK-LABEL: func @producer_has_another_consumer
// CHECK-NOT:     bufferization.alloc_tensor
func.func @producer_has_another_consumer(%acc: tensor<1x16x16x8xi32>, %f: tensor<3x3x8x8xi8>)
    -> (tensor<1x16x16x8xi32>, tensor<1x16x16x8xi8>) {
  %zero = arith.constant 0 : i8
  %z = arith.constant 0 : i32
  %s = arith.constant 2.000000e-02 : f32
  %e = tensor.empty() : tensor<1x16x16x8xi8>
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%acc : tensor<1x16x16x8xi32>) outs(%e : tensor<1x16x16x8xi8>) {
  ^bb0(%a: i32, %b: i8):
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    %r = arith.fptosi %m : f32 to i8
    linalg.yield %r : i8
  } -> tensor<1x16x16x8xi8>
  %padded = tensor.pad %q low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%i: index, %j: index, %k: index, %l: index):
    tensor.yield %zero : i8
  } : tensor<1x16x16x8xi8> to tensor<1x18x18x8xi8>
  %oe = tensor.empty() : tensor<1x16x16x8xi32>
  %oz = linalg.fill ins(%z : i32) outs(%oe : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%padded, %f : tensor<1x18x18x8xi8>, tensor<3x3x8x8xi8>)
       outs(%oz : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  // the shortcut, which is the other consumer
  return %c, %q : tensor<1x16x16x8xi32>, tensor<1x16x16x8xi8>
}

// A grouped convolution reaches its padding through one `tensor.extract_slice`
// per group, so the convolution that could take the padding is two steps below
// it. Without looking through the slice the whole grouped family kept its
// pointwise convolution in software: its requantization was writing into the
// padded buffer's middle, and the accelerator cannot address that window --
// `out_stride` is one stride per pixel, and the rows of a window narrower than
// its buffer do not follow it. `grp` went from four offloaded convolutions to
// five.
// CHECK-LABEL: func @grouped_reaches_through_a_slice
// CHECK:         %[[Q:.*]] = linalg.generic
// CHECK:         %[[OWN:.*]] = bufferization.alloc_tensor() copy(%[[Q]])
// CHECK:         tensor.pad %[[OWN]]
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @grouped_reaches_through_a_slice(%acc: tensor<1x16x16x32xi32>, %f: tensor<3x3x8x8xi8>)
    -> tensor<1x16x16x8xi32> {
  %zero = arith.constant 0 : i8
  %z = arith.constant 0 : i32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x16x16x32xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
       ins(%acc : tensor<1x16x16x32xi32>) outs(%e : tensor<1x16x16x32xi8>) {
  ^bb0(%a: i32, %b: i8):
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    %r = arith.fptosi %m : f32 to i32
    %c1 = arith.maxsi %r, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x16x16x32xi8>
  %padded = tensor.pad %q low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : i8
  } : tensor<1x16x16x32xi8> to tensor<1x18x18x32xi8>
  %g = tensor.extract_slice %padded[0, 0, 0, 0] [1, 18, 18, 8] [1, 1, 1, 1]
    : tensor<1x18x18x32xi8> to tensor<1x18x18x8xi8>
  %o = tensor.empty() : tensor<1x16x16x8xi32>
  %fl = linalg.fill ins(%z : i32) outs(%o : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%g, %f : tensor<1x18x18x8xi8>, tensor<3x3x8x8xi8>) outs(%fl : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  return %c : tensor<1x16x16x8xi32>
}

// And when the groups have already become im2col packs there is no convolution
// below the padding at all -- but the one *above* it is still being kept in
// software for the same reason. `gup` went from four offloaded operations to
// five.
// CHECK-LABEL: func @foldable_producer_above
// CHECK:         %[[Q2:.*]] = linalg.generic
// CHECK:         %[[OWN2:.*]] = bufferization.alloc_tensor() copy(%[[Q2]])
// CHECK:         tensor.pad %[[OWN2]]
func.func @foldable_producer_above(%in: tensor<1x16x16x16xi8>, %w: tensor<1x1x16x32xi8>)
    -> tensor<1x18x18x32xi8> {
  %zero = arith.constant 0 : i8
  %z = arith.constant 0 : i32
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %o = tensor.empty() : tensor<1x16x16x32xi32>
  %fl = linalg.fill ins(%z : i32) outs(%o : tensor<1x16x16x32xi32>) -> tensor<1x16x16x32xi32>
  %acc = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %w : tensor<1x16x16x16xi8>, tensor<1x1x16x32xi8>) outs(%fl : tensor<1x16x16x32xi32>) -> tensor<1x16x16x32xi32>
  %e = tensor.empty() : tensor<1x16x16x32xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
       ins(%acc : tensor<1x16x16x32xi32>) outs(%e : tensor<1x16x16x32xi8>) {
  ^bb0(%a: i32, %b: i8):
    %f1 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f1, %s : f32
    %r = arith.fptosi %m : f32 to i32
    %c1 = arith.maxsi %r, %lo : i32
    %c2 = arith.minsi %c1, %hi : i32
    %t = arith.trunci %c2 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x16x16x32xi8>
  %padded = tensor.pad %q low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : i8
  } : tensor<1x16x16x32xi8> to tensor<1x18x18x32xi8>
  return %padded : tensor<1x18x18x32xi8>
}
