// A convolution reaches the accelerator as a matmul. This sidesteps the layout
// problem instead of solving it: no frontend hands over the layout the runtime
// wants -- PyTorch lowers to conv_2d_nchw_fchw, TOSA to conv_2d_nhwc_fhwc, and
// tiled_conv_auto reads (KH, KW, C, F) -- but after im2col there is only a
// matmul left.

// RUN: gemmlir-opt --conv-to-img2col --canonicalize %s | FileCheck %s
// RUN: gemmlir-opt --conv-to-img2col --canonicalize %s | FileCheck %s --check-prefix=SPLIT
// RUN: gemmlir-opt --conv-to-img2col=unfoldable-only=1 --canonicalize %s | FileCheck %s --check-prefix=ONLY

// The convolution is gone, replaced by an im2col pack and a contraction.
// With a batch of 1 the contraction comes out as a generic rather than a named
// matmul, which is why gemmlir does not pick it up yet -- see docs/pipeline.md.
// CHECK-LABEL: func.func @nhwc
// CHECK-NOT:     linalg.conv_2d_nhwc_hwcf
// CHECK:         linalg.generic
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @nhwc(%in: tensor<1x16x16x4xf32>, %f: tensor<3x3x4x8xf32>,
                %out: tensor<1x14x14x8xf32>) -> tensor<1x14x14x8xf32> {
  %r = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %f : tensor<1x16x16x4xf32>, tensor<3x3x4x8xf32>)
    outs(%out : tensor<1x14x14x8xf32>) -> tensor<1x14x14x8xf32>
  return %r : tensor<1x14x14x8xf32>
}

// PyTorch's NCHW form is handled too.
// CHECK-LABEL: func.func @nchw
// CHECK-NOT:     linalg.conv_2d_nchw_fchw
// CHECK:         linalg.generic
func.func @nchw(%in: tensor<1x4x16x16xf32>, %f: tensor<8x4x3x3xf32>,
                %out: tensor<1x8x14x14xf32>) -> tensor<1x8x14x14xf32> {
  %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %f : tensor<1x4x16x16xf32>, tensor<8x4x3x3xf32>)
    outs(%out : tensor<1x8x14x14xf32>) -> tensor<1x8x14x14xf32>
  return %r : tensor<1x8x14x14xf32>
}

// For NHWC the pack is written as nested loops over the patch offset instead of
// the collapsed `K` that MLIR's rewrite produces. Taking a collapsed offset
// apart costs a floordiv and a mod per level, and NHWC puts that index on the
// *innermost* loop, so the divisors -- the kernel's, never powers of two -- run
// on every packed element. Measured on the board for the CNN's first pack:
// 9.21 ms collapsed, 4.13 ms iterated position-innermost, 1.77 ms like this,
// against 3.79 ms for the NCHW pack it replaces. On the whole CNN it was
// 13.44 ms to 5.39 ms.

// The gather's read map is pure addition: the loop indices *are* kh, kw and c.
// SPLIT-DAG:   #[[READ:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d3, d2 * 2 + d4, d5)>
// SPLIT-LABEL: func.func @nhwc_split_pack
// SPLIT:         %[[P:.*]] = linalg.generic
// SPLIT-SAME:      indexing_maps = [#[[READ]], #{{.*}}]
// SPLIT-SAME:      ins(%arg0 : tensor<1x9x9x2xf32>)
// SPLIT-SAME:      outs(%{{.*}} : tensor<1x4x4x3x3x2xf32>)
// SPLIT-NEXT:    ^bb0(%[[V:.*]]: f32, %{{.*}}: f32):
// SPLIT-NEXT:      linalg.yield %[[V]]
// SPLIT:         %[[C:.*]] = tensor.collapse_shape %[[P]] {{\[}}[0, 1, 2], [3, 4, 5]] {{.*}} into tensor<16x18xf32>
// SPLIT:         linalg.matmul ins(%[[C]], %{{.*}} : tensor<16x18xf32>, tensor<18x4xf32>)
func.func @nhwc_split_pack(%in: tensor<1x9x9x2xf32>, %f: tensor<3x3x2x4xf32>,
                           %init: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%in, %f : tensor<1x9x9x2xf32>, tensor<3x3x2x4xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %c : tensor<1x4x4x4xf32>
}

// A batch of more than one falls through to MLIR's own rewrite, which is
// correct and general; only the single-image case is special-cased here. That
// rewrite packs into three dimensions, so the patch offset is collapsed again.
// SPLIT-LABEL: func.func @batched_falls_through
// SPLIT:         linalg.generic
// SPLIT-SAME:      outs(%{{.*}} : tensor<2x16x18xf32>)
func.func @batched_falls_through(%in: tensor<2x9x9x2xf32>, %f: tensor<3x3x2x4xf32>,
                                 %init: tensor<2x4x4x4xf32>) -> tensor<2x4x4x4xf32> {
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%in, %f : tensor<2x9x9x2xf32>, tensor<3x3x2x4xf32>) outs(%init : tensor<2x4x4x4xf32>) -> tensor<2x4x4x4xf32>
  return %c : tensor<2x4x4x4xf32>
}

// `tiled_conv_auto` writes `elem_t` and there is no other form, so a
// convolution whose result the model returns wide has nothing to fold into and
// stays a scalar loop -- the whole grouped family had no accelerator call for
// its convolutions at all. Packed first it becomes a matmul, and the matmul
// call does write the i32 accumulator. Measured on the board: ResNeXt's bare
// grouped convolution 259.8 -> 51.3 ms, and that is with the pack.
// ONLY-LABEL: func.func @unfoldable_is_packed
// ONLY:         linalg.matmul
func.func @unfoldable_is_packed(%in: tensor<1x9x9x2xi8>, %f: tensor<3x3x2x4xi8>,
                                %init: tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xf32> {
  %s = arith.constant 2.000000e-02 : f32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%in, %f : tensor<1x9x9x2xi8>, tensor<3x3x2x4xi8>) outs(%init : tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi32>
  %e = tensor.empty() : tensor<1x4x4x4xf32>
  %d = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x4x4x4xi32>) outs(%e : tensor<1x4x4x4xf32>) {
  ^bb0(%v: i32, %o: f32):
    %f32 = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f32, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x4x4x4xf32>
  return %d : tensor<1x4x4x4xf32>
}

// One that would fold is left alone, however far away the requantization is:
// a residual add and a global pool sit between a ResNet block's last
// convolution and the quantization of the block's output, and
// `--convert-linalg-to-gemmlir` reads that whole shape. Packing it cost a
// `conv2d_i8` and 23000 scalar elements on `gapb`.
// ONLY-LABEL: func.func @foldable_is_left_alone
// ONLY:         linalg.conv_2d_nhwc_hwcf
// ONLY-NOT:     linalg.matmul
func.func @foldable_is_left_alone(%in: tensor<1x9x9x2xi8>, %f: tensor<3x3x2x4xi8>,
                                  %init: tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%in, %f : tensor<1x9x9x2xi8>, tensor<3x3x2x4xi8>) outs(%init : tensor<1x4x4x4xi32>) -> tensor<1x4x4x4xi32>
  %e = tensor.empty() : tensor<1x4x4x4xf32>
  %d = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x4x4x4xi32>) outs(%e : tensor<1x4x4x4xf32>) {
  ^bb0(%v: i32, %o: f32):
    %f32 = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f32, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x4x4x4xf32>
  %e2 = tensor.empty() : tensor<1x4x4x4xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%d : tensor<1x4x4x4xf32>) outs(%e2 : tensor<1x4x4x4xi8>) {
  ^bb0(%v: f32, %o: i8):
    %i = arith.fptosi %v : f32 to i8
    linalg.yield %i : i8
  } -> tensor<1x4x4x4xi8>
  return %q : tensor<1x4x4x4xi8>
}

// An f32 convolution is not one of these either: there is no accelerator call
// for it at all, and the pack would be pure loss.
// ONLY-LABEL: func.func @float_is_left_alone
// ONLY:         linalg.conv_2d_nhwc_hwcf
// ONLY-NOT:     linalg.matmul
func.func @float_is_left_alone(%in: tensor<1x9x9x2xf32>, %f: tensor<3x3x2x4xf32>,
                               %init: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%in, %f : tensor<1x9x9x2xf32>, tensor<3x3x2x4xf32>) outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %c : tensor<1x4x4x4xf32>
}

// One shape the accelerator computes wrong goes this way whether or not it
// could have folded: a 3x3 stride-1 convolution with a 24x24 result and at most
// five input channels writes one output pixel -- (22, 23), every channel of it
// -- with plausible but wrong values. Measured on the board against an exact
// integer reference, one convolution per process; `conv_cpu` is exact on all
// 488 shapes swept and so is every other size from 4 to 64. It kept `atr` and
// `atrn` wrong for weeks at an unchanged relative L2.
// ONLY-LABEL: func.func @broken_shape_is_packed
// ONLY:         linalg.matmul
func.func @broken_shape_is_packed(%in: tensor<1x24x24x3xi8>, %f: tensor<3x3x3x16xi8>,
                                  %init: tensor<1x24x24x16xi32>) -> tensor<1x24x24x16xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %p = tensor.pad %in low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    %z = arith.constant 0 : i8
    tensor.yield %z : i8
  } : tensor<1x24x24x3xi8> to tensor<1x26x26x3xi8>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%p, %f : tensor<1x26x26x3xi8>, tensor<3x3x3x16xi8>) outs(%init : tensor<1x24x24x16xi32>) -> tensor<1x24x24x16xi32>
  %e = tensor.empty() : tensor<1x24x24x16xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x24x24x16xi32>) outs(%e : tensor<1x24x24x16xi8>) {
  ^bb0(%v: i32, %o: i8):
    %f32 = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f32, %s : f32
    %i = arith.fptosi %m : f32 to i8
    linalg.yield %i : i8
  } -> tensor<1x24x24x16xi8>
  return %q : tensor<1x24x24x16xi8>
}

// Six input channels at the same size is not that shape, and a convolution that
// folds stays a convolution.
// ONLY-LABEL: func.func @six_channels_is_left_alone
// ONLY:         linalg.conv_2d_nhwc_hwcf
// ONLY-NOT:     linalg.matmul
func.func @six_channels_is_left_alone(%in: tensor<1x26x26x6xi8>, %f: tensor<3x3x6x16xi8>,
                                      %init: tensor<1x24x24x16xi32>) -> tensor<1x24x24x16xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %f : tensor<1x26x26x6xi8>, tensor<3x3x6x16xi8>) outs(%init : tensor<1x24x24x16xi32>) -> tensor<1x24x24x16xi32>
  %e = tensor.empty() : tensor<1x24x24x16xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x24x24x16xi32>) outs(%e : tensor<1x24x24x16xi8>) {
  ^bb0(%v: i32, %o: i8):
    %f32 = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f32, %s : f32
    %i = arith.fptosi %m : f32 to i8
    linalg.yield %i : i8
  } -> tensor<1x24x24x16xi8>
  return %q : tensor<1x24x24x16xi8>
}

// A join while the values are still wide is as final as the return. `conv2d_i8`
// writes one window of one buffer; it cannot write a branch's slice of a
// concatenation in f32, so the tail below such a convolution is never going to
// become a requantization it can take. ShuffleNet's unit is exactly that --
// its two branches are joined in f32 and quantized only after the channel
// shuffle -- and its 3x3 branch was the last convolution in the model set still
// running as a scalar loop, worth 56.6 ms of `shub`.
// ONLY-LABEL: func.func @joined_wide_is_packed
// ONLY:         linalg.matmul
func.func @joined_wide_is_packed(%in: tensor<1x18x18x8xi8>, %f: tensor<3x3x8x8xi8>,
                                 %init: tensor<1x16x16x8xi32>, %other: tensor<1x16x16x8xf32>)
    -> tensor<1x16x16x16xf32> {
  %s = arith.constant 2.000000e-02 : f32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %f : tensor<1x18x18x8xi8>, tensor<3x3x8x8xi8>) outs(%init : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %e = tensor.empty() : tensor<1x16x16x8xf32>
  %d = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x16x16x8xi32>) outs(%e : tensor<1x16x16x8xf32>) {
  ^bb0(%v: i32, %o: f32):
    %f32 = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f32, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x16x16x8xf32>
  %j = tensor.concat dim(3) %d, %other : (tensor<1x16x16x8xf32>, tensor<1x16x16x8xf32>) -> tensor<1x16x16x16xf32>
  return %j : tensor<1x16x16x16xf32>
}

// A join the convolution reaches only *after* its requantization is a different
// thing: the convolution folds and writes the branch's slice itself, through
// `out_stride`.
// ONLY-LABEL: func.func @joined_narrow_is_left_alone
// ONLY:         linalg.conv_2d_nhwc_hwcf
// ONLY-NOT:     linalg.matmul
func.func @joined_narrow_is_left_alone(%in: tensor<1x18x18x8xi8>, %f: tensor<3x3x8x8xi8>,
                                       %init: tensor<1x16x16x8xi32>, %other: tensor<1x16x16x8xi8>)
    -> tensor<1x16x16x16xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %f : tensor<1x18x18x8xi8>, tensor<3x3x8x8xi8>) outs(%init : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %e = tensor.empty() : tensor<1x16x16x8xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x16x16x8xi32>) outs(%e : tensor<1x16x16x8xi8>) {
  ^bb0(%v: i32, %o: i8):
    %f32 = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f32, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x16x16x8xi8>
  %j = tensor.concat dim(3) %q, %other : (tensor<1x16x16x8xi8>, tensor<1x16x16x8xi8>) -> tensor<1x16x16x16xi8>
  return %j : tensor<1x16x16x16xi8>
}

// `tiled_conv_auto` takes one integer for the kernel, one for the stride and
// one for the dilation, so `Conv2DInt8Op`'s verifier refuses a convolution
// whose two spatial extents differ. Such a convolution has no accelerator call
// to fold into and would stay a scalar loop over every multiply -- a separable
// 3x3 written as a 1x3 and a 3x1 is 67,600 of them on a 16x16 image. A matmul
// has no such restriction, so pack it even though its result is requantized.

// ONLY-LABEL: func.func @kernel_not_square
// ONLY:         linalg.matmul
// ONLY-NOT:     linalg.conv_2d_nhwc_hwcf
func.func @kernel_not_square(%in: tensor<1x16x18x3xi8>, %f: tensor<1x3x3x8xi8>,
                             %init: tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32> {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>,
                                 strides = dense<1> : vector<2xi64>}
    ins(%in, %f : tensor<1x16x18x3xi8>, tensor<1x3x3x8xi8>)
    outs(%init : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  return %0 : tensor<1x16x16x8xi32>
}

// A stride the accelerator cannot say either.
// ONLY-LABEL: func.func @stride_not_square
// ONLY:         linalg.matmul
// ONLY-NOT:     linalg.conv_2d_nhwc_hwcf
func.func @stride_not_square(%in: tensor<1x17x18x3xi8>, %f: tensor<3x3x3x8xi8>,
                             %init: tensor<1x8x16x8xi32>) -> tensor<1x8x16x8xi32> {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>,
                                 strides = dense<[2, 1]> : vector<2xi64>}
    ins(%in, %f : tensor<1x17x18x3xi8>, tensor<3x3x3x8xi8>)
    outs(%init : tensor<1x8x16x8xi32>) -> tensor<1x8x16x8xi32>
  return %0 : tensor<1x8x16x8xi32>
}

// A square one whose result is requantized still goes the convolution way:
// `conv2d_i8` takes the bias, the activation and the pooling with it, and the
// pack would be pure cost.
// ONLY-LABEL: func.func @square_is_left_alone
// ONLY:         linalg.conv_2d_nhwc_hwcf
// ONLY-NOT:     linalg.matmul
func.func @square_is_left_alone(%in: tensor<1x18x18x3xi8>, %f: tensor<3x3x3x8xi8>,
                                %init: tensor<1x16x16x8xi32>, %out: tensor<1x16x16x8xi8>)
    -> tensor<1x16x16x8xi8> {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %s = arith.constant 5.000000e-02 : f32
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>,
                                 strides = dense<1> : vector<2xi64>}
    ins(%in, %f : tensor<1x18x18x3xi8>, tensor<3x3x3x8xi8>)
    outs(%init : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %1 = linalg.generic {indexing_maps = [#id4, #id4],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%0 : tensor<1x16x16x8xi32>) outs(%out : tensor<1x16x16x8xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f32 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f32, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %c = arith.maxsi %i, %lo : i32
    %d = arith.minsi %c, %hi : i32
    %t = arith.trunci %d : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x16x16x8xi8>
  return %1 : tensor<1x16x16x8xi8>
}

// The accelerator writes exactly its own output, so a requantization that sits
// below something which made *more* elements is not one this convolution can
// take. A decoder's nearest-neighbour upsample is the case: `conv ->
// dequantize -> upsample -> quantize` ends in an i8, and taking that as proof
// the convolution folds left a stride-2 convolution as a scalar loop while
// every other layer of the block was on the accelerator.

#up6 = affine_map<(d0,d1,d2,d3,d4,d5) -> (d0,d1,d2,d4)>
#id6 = affine_map<(d0,d1,d2,d3,d4,d5) -> (d0,d1,d2,d3,d4,d5)>

// ONLY-LABEL: func.func @requantize_below_a_growth
// ONLY:         linalg.matmul
// ONLY-NOT:     linalg.conv_2d_nhwc_hwcf
func.func @requantize_below_a_growth(%in: tensor<1x18x18x8xi8>, %f: tensor<3x3x8x16xi8>,
                                     %init: tensor<1x8x8x16xi32>,
                                     %wide: tensor<1x8x8x2x16x2xf32>,
                                     %out: tensor<1x8x8x2x16x2xi8>)
    -> tensor<1x8x8x2x16x2xi8> {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %s = arith.constant 5.000000e-02 : f32
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>,
                                 strides = dense<2> : vector<2xi64>}
    ins(%in, %f : tensor<1x18x18x8xi8>, tensor<3x3x8x16xi8>)
    outs(%init : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  // the upsample: every element read four times
  %1 = linalg.generic {indexing_maps = [#up6, #id6],
                       iterator_types = ["parallel","parallel","parallel","parallel","parallel","parallel"]}
    ins(%0 : tensor<1x8x8x16xi32>) outs(%wide : tensor<1x8x8x2x16x2xf32>) {
  ^bb0(%a: i32, %o: f32):
    %f32 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f32, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x8x8x2x16x2xf32>
  %2 = linalg.generic {indexing_maps = [#id6, #id6],
                       iterator_types = ["parallel","parallel","parallel","parallel","parallel","parallel"]}
    ins(%1 : tensor<1x8x8x2x16x2xf32>) outs(%out : tensor<1x8x8x2x16x2xi8>) {
  ^bb0(%v: f32, %o: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c = arith.maxsi %i, %lo : i32
    %e = arith.minsi %c, %hi : i32
    %t = arith.trunci %e : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x8x8x2x16x2xi8>
  return %2 : tensor<1x8x8x2x16x2xi8>
}

// A residual add keeps the element count and a global pool lowers it, so
// neither makes the requantization somebody else's -- this one still folds.
#id4b = affine_map<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>

// ONLY-LABEL: func.func @same_size_on_the_way_down
// ONLY:         linalg.conv_2d_nhwc_hwcf
// ONLY-NOT:     linalg.matmul
func.func @same_size_on_the_way_down(%in: tensor<1x18x18x8xi8>, %f: tensor<3x3x8x16xi8>,
                                     %init: tensor<1x16x16x16xi32>,
                                     %mid: tensor<1x16x16x16xf32>,
                                     %out: tensor<1x16x16x16xi8>) -> tensor<1x16x16x16xi8> {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %s = arith.constant 5.000000e-02 : f32
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>,
                                 strides = dense<1> : vector<2xi64>}
    ins(%in, %f : tensor<1x18x18x8xi8>, tensor<3x3x8x16xi8>)
    outs(%init : tensor<1x16x16x16xi32>) -> tensor<1x16x16x16xi32>
  %1 = linalg.generic {indexing_maps = [#id4b, #id4b],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%0 : tensor<1x16x16x16xi32>) outs(%mid : tensor<1x16x16x16xf32>) {
  ^bb0(%a: i32, %o: f32):
    %f32 = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f32, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x16x16x16xf32>
  %2 = linalg.generic {indexing_maps = [#id4b, #id4b],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%1 : tensor<1x16x16x16xf32>) outs(%out : tensor<1x16x16x16xi8>) {
  ^bb0(%v: f32, %o: i8):
    %d = arith.divf %v, %s : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c = arith.maxsi %i, %lo : i32
    %e = arith.minsi %c, %hi : i32
    %t = arith.trunci %e : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x16x16x16xi8>
  return %2 : tensor<1x16x16x16xi8>
}

// A `conv2d_i8` writes i8, so its tail has to be something the mvout pipeline
// can do: `saturate(scale * accumulator + bias)` with at most a relu. A
// transcendental in the way means there is no requantization to fold into,
// however much the i8 below looks like one -- and the convolution belongs on
// the matmul path instead, where `matmul_i8` leaves an i32 accumulator for the
// tail to read.
//
// EfficientNet is the case: **SiLU**, `x * sigmoid(x)`, sits between every
// convolution and its quantization, and thirty-three convolutions stayed scalar
// loops because the walk read the i8 under the SiLU as proof they folded.

#id4c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// ONLY-LABEL: func.func @silu_tail_is_not_a_requantize
// ONLY:         linalg.generic
// ONLY:         linalg.matmul
// ONLY-NOT:     linalg.conv_2d_nhwc_hwcf
func.func @silu_tail_is_not_a_requantize(%in: tensor<1x8x8x16xi8>, %flt: tensor<3x3x16x16xi8>)
    -> tensor<1x6x6x16xi8> {
  %z = arith.constant 0 : i32
  %one = arith.constant 1.000000e+00 : f32
  %s = arith.constant 2.500000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x6x6x16xi32>
  %init = linalg.fill ins(%z : i32) outs(%e : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt : tensor<1x8x8x16xi8>, tensor<3x3x16x16xi8>)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  %o = tensor.empty() : tensor<1x6x6x16xi8>
  %q = linalg.generic {indexing_maps = [#id4c, #id4c],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%c : tensor<1x6x6x16xi32>) outs(%o : tensor<1x6x6x16xi8>) {
  ^bb0(%a: i32, %out: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    // SiLU: x / (1 + exp(-x))
    %n = arith.negf %m : f32
    %x = math.exp %n : f32
    %d = arith.addf %x, %one : f32
    %v = arith.divf %m, %d : f32
    %r = math.roundeven %v : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x6x6x16xi8>
  return %q : tensor<1x6x6x16xi8>
}

// The same convolution with a plain requantization tail -- scale, relu,
// saturate -- is one the accelerator's own pipeline does, so it stays a
// convolution.

#id4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// ONLY-LABEL: func.func @plain_requantize_tail_stays
// ONLY:         linalg.conv_2d_nhwc_hwcf
// ONLY-NOT:     linalg.matmul
func.func @plain_requantize_tail_stays(%in: tensor<1x8x8x16xi8>, %flt: tensor<3x3x16x16xi8>)
    -> tensor<1x6x6x16xi8> {
  %z = arith.constant 0 : i32
  %zero = arith.constant 0.000000e+00 : f32
  %s = arith.constant 2.500000e-02 : f32
  %t2 = arith.constant 5.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %e = tensor.empty() : tensor<1x6x6x16xi32>
  %init = linalg.fill ins(%z : i32) outs(%e : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt : tensor<1x8x8x16xi8>, tensor<3x3x16x16xi8>)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  %o = tensor.empty() : tensor<1x6x6x16xi8>
  %q = linalg.generic {indexing_maps = [#id4d, #id4d],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%c : tensor<1x6x6x16xi32>) outs(%o : tensor<1x6x6x16xi8>) {
  ^bb0(%a: i32, %out: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %cm = arith.cmpf ugt, %m, %zero : f32
    %r0 = arith.select %cm, %m, %zero : f32
    %d = arith.divf %r0, %t2 : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %tr = arith.trunci %c1 : i32 to i8
    linalg.yield %tr : i8
  } -> tensor<1x6x6x16xi8>
  return %q : tensor<1x6x6x16xi8>
}

// Tried and reverted, with the measurement: calling a **reduction** below the
// convolution unabsorbable -- which it is, the accelerator writes exactly its
// own output and a layer norm's mean does not have that shape -- packs this
// convolution as a matmul. On the board that is the wrong trade: it moves two
// of EfficientNet's convolutions off `conv2d_i8` and costs **1114 -> 1373 ms**,
// while ConvNeXt's patchify convolution is rescued by --quantize-unfoldable-
// tails rather than by packing. So the walk deliberately judges a reduction
// elsewhere, and this convolution stays a convolution.

#id4e = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#drop = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// ONLY-LABEL: func.func @a_layer_norm_below_is_final
// ONLY:         linalg.conv_2d_nhwc_hwcf
func.func @a_layer_norm_below_is_final(%in: tensor<1x8x8x16xi8>, %flt: tensor<3x3x16x16xi8>)
    -> (tensor<1x6x6x16xf32>, tensor<1x6x6xf32>) {
  %z = arith.constant 0 : i32
  %zero = arith.constant 0.000000e+00 : f32
  %s = arith.constant 2.500000e-02 : f32
  %e = tensor.empty() : tensor<1x6x6x16xi32>
  %init = linalg.fill ins(%z : i32) outs(%e : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt : tensor<1x8x8x16xi8>, tensor<3x3x16x16xi8>)
    outs(%init : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  %o = tensor.empty() : tensor<1x6x6x16xf32>
  %d = linalg.generic {indexing_maps = [#id4e, #id4e],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%c : tensor<1x6x6x16xi32>) outs(%o : tensor<1x6x6x16xf32>) {
  ^bb0(%a: i32, %x: f32):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x6x6x16xf32>
  // the mean, reading the value the subtraction also reads
  %me = tensor.empty() : tensor<1x6x6xf32>
  %mi = linalg.fill ins(%zero : f32) outs(%me : tensor<1x6x6xf32>) -> tensor<1x6x6xf32>
  %mean = linalg.generic {indexing_maps = [#id4e, #drop],
                          iterator_types = ["parallel","parallel","parallel","reduction"]}
      ins(%d : tensor<1x6x6x16xf32>) outs(%mi : tensor<1x6x6xf32>) {
  ^bb0(%a: f32, %x: f32):
    %t = arith.addf %a, %x : f32
    linalg.yield %t : f32
  } -> tensor<1x6x6xf32>
  return %d, %mean : tensor<1x6x6x16xf32>, tensor<1x6x6xf32>
}

// A **pool** while the values are still wide is as final as a join, and for the
// same reason: `conv2d_i8` writes its own output buffer, so it cannot produce a
// pooled one. `FoldMaxPoolIntoConv` does attach a pool to a call, but only to a
// call that already exists -- which needs the tail to have become a
// requantization first, and a tail still in f32 *here* is one that never will.
//
// DenseNet-121's stem is that shape: a 7x7 stride-2 convolution on three
// channels whose f32 tail feeds a max-pool. The pooled result has seven users,
// so the branch test stopped the walk one step too late and the convolution
// stayed a scalar loop -- **72% of the whole model**, by program-counter
// sampling. Three input channels is also next door to the corner the
// accelerator gets wrong, and im2col plus a matmul is the route that agrees
// with the host.
// ONLY-LABEL: func.func @pooled_wide_is_packed
// ONLY:         linalg.matmul
func.func @pooled_wide_is_packed(%in: tensor<1x18x18x8xi8>, %f: tensor<3x3x8x8xi8>,
                                 %init: tensor<1x16x16x8xi32>, %w: tensor<3x3xf32>,
                                 %po: tensor<1x14x14x8xf32>) -> tensor<1x14x14x8xf32> {
  %s = arith.constant 2.000000e-02 : f32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %f : tensor<1x18x18x8xi8>, tensor<3x3x8x8xi8>) outs(%init : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %e = tensor.empty() : tensor<1x16x16x8xf32>
  %d = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x16x16x8xi32>) outs(%e : tensor<1x16x16x8xf32>) {
  ^bb0(%v: i32, %o: f32):
    %f32 = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f32, %s : f32
    linalg.yield %m : f32
  } -> tensor<1x16x16x8xf32>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%d, %w : tensor<1x16x16x8xf32>, tensor<3x3xf32>) outs(%po : tensor<1x14x14x8xf32>) -> tensor<1x14x14x8xf32>
  return %p : tensor<1x14x14x8xf32>
}

// A pool the convolution reaches only *after* its requantization is a different
// thing -- that is the one `FoldMaxPoolIntoConv` folds into the call, and
// packing it would throw away both the call and the fused pool.
// ONLY-LABEL: func.func @pooled_narrow_is_left_alone
// ONLY:         linalg.conv_2d_nhwc_hwcf
// ONLY-NOT:     linalg.matmul
func.func @pooled_narrow_is_left_alone(%in: tensor<1x18x18x8xi8>, %f: tensor<3x3x8x8xi8>,
                                       %init: tensor<1x16x16x8xi32>, %w: tensor<3x3xi8>,
                                       %po: tensor<1x14x14x8xi8>) -> tensor<1x14x14x8xi8> {
  %s = arith.constant 2.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %f : tensor<1x18x18x8xi8>, tensor<3x3x8x8xi8>) outs(%init : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %e = tensor.empty() : tensor<1x16x16x8xi8>
  %q = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%c : tensor<1x16x16x8xi32>) outs(%e : tensor<1x16x16x8xi8>) {
  ^bb0(%v: i32, %o: i8):
    %f32 = arith.sitofp %v : i32 to f32
    %m = arith.mulf %f32, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x16x16x8xi8>
  %p = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%q, %w : tensor<1x16x16x8xi8>, tensor<3x3xi8>) outs(%po : tensor<1x14x14x8xi8>) -> tensor<1x14x14x8xi8>
  return %p : tensor<1x14x14x8xi8>
}
