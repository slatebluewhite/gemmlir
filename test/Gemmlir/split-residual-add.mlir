// In a residual block the second convolution's tail *is* the add: one
// linalg.generic reads the shortcut, the accumulator and the bias and writes
// the block's i8 output. --convert-linalg-to-gemmlir reads a requantization of
// exactly one accumulator, so it cannot fold that, and the convolution stays a
// scalar loop.

// RUN: gemmlir-opt --split-residual-add --canonicalize %s | FileCheck %s
// RUN: gemmlir-opt --split-residual-add --canonicalize \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
// RUN:   --buffer-deallocation-pipeline --convert-linalg-to-gemmlir %s \
// RUN: | FileCheck %s --check-prefix=FOLDED

#nhwc = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>

// The convolution's own requantization comes out into its own operation, with
// the accumulator first and the bias second -- the shape the conversion reads.
// The intermediate carries the block output's scale (0.02), so the scale on the
// convolution is the accumulator's over that one.
// CHECK-LABEL: func.func @residual
// CHECK:         %[[T:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[ACC:.*]], %[[BIAS:.*]] : tensor<1x16x16x8xi32>, tensor<8xi32>)
// CHECK:           arith.addi
// CHECK:           %[[W:.*]] = arith.sitofp
// CHECK:           arith.mulf %[[W]], %{{.*}}
// CHECK:           arith.trunci
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %[[T]] : tensor<1x16x16x8xi8>, tensor<1x16x16x8xi8>)
// CHECK:           arith.trunci

// And with that, the convolution folds into one call and the add into another.
// FOLDED-LABEL: func.func @residual
// FOLDED:         gemmlir.conv2d_i8
// FOLDED-SAME:      bias(
// FOLDED-SAME:      scale = 0.0250000022
// FOLDED:         gemmlir.resadd_i8
// FOLDED-SAME:      lhs_scale = 5.000000e-01
// FOLDED-NOT:     linalg.conv_2d
func.func @residual(%in: tensor<1x18x18x8xi8>, %w: tensor<3x3x8x8xi8>,
                    %bias: tensor<8xi32>, %shortcut: tensor<1x16x16x8xi8>)
    -> tensor<1x16x16x8xi8> {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : i32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %accScale = arith.constant 5.000000e-04 : f32
  %shortScale = arith.constant 1.000000e-02 : f32
  %outScale = arith.constant 2.000000e-02 : f32

  %e = tensor.empty() : tensor<1x16x16x8xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>
  %acc = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : tensor<1x18x18x8xi8>, tensor<3x3x8x8xi8>)
    outs(%f : tensor<1x16x16x8xi32>) -> tensor<1x16x16x8xi32>

  %o = tensor.empty() : tensor<1x16x16x8xi8>
  %r = linalg.generic {indexing_maps = [#nhwc, #nhwc, #chan, #id],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%shortcut, %acc, %bias : tensor<1x16x16x8xi8>, tensor<1x16x16x8xi32>, tensor<8xi32>)
    outs(%o : tensor<1x16x16x8xi8>) {
  ^bb0(%s: i8, %a: i32, %b: i32, %out: i8):
    %sum = arith.addi %a, %b : i32
    %sf = arith.sitofp %s : i8 to f32
    %af = arith.sitofp %sum : i32 to f32
    %sm = arith.mulf %sf, %shortScale : f32
    %am = arith.mulf %af, %accScale : f32
    %add = arith.addf %sm, %am : f32
    %gt = arith.cmpf ugt, %add, %zero : f32
    %relu = arith.select %gt, %add, %zero : f32
    %q = arith.divf %relu, %outScale : f32
    %rd = math.roundeven %q : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x16x16x8xi8>
  return %r : tensor<1x16x16x8xi8>
}

// An accumulator that no accelerator call produces has nothing to uncover, and
// splitting it would only add a rounding.
// CHECK-LABEL: func.func @not_a_contraction
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<1x4x4x2xi8>, tensor<1x4x4x2xi32>)
// CHECK-NOT:     linalg.generic
func.func @not_a_contraction(%shortcut: tensor<1x4x4x2xi8>, %acc: tensor<1x4x4x2xi32>)
    -> tensor<1x4x4x2xi8> {
  %zero = arith.constant 0.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %accScale = arith.constant 5.000000e-04 : f32
  %shortScale = arith.constant 1.000000e-02 : f32
  %outScale = arith.constant 2.000000e-02 : f32
  %o = tensor.empty() : tensor<1x4x4x2xi8>
  %r = linalg.generic {indexing_maps = [#nhwc, #nhwc, #id],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%shortcut, %acc : tensor<1x4x4x2xi8>, tensor<1x4x4x2xi32>)
    outs(%o : tensor<1x4x4x2xi8>) {
  ^bb0(%s: i8, %a: i32, %out: i8):
    %sf = arith.sitofp %s : i8 to f32
    %af = arith.sitofp %a : i32 to f32
    %sm = arith.mulf %sf, %shortScale : f32
    %am = arith.mulf %af, %accScale : f32
    %add = arith.addf %sm, %am : f32
    %q = arith.divf %add, %outScale : f32
    %rd = math.roundeven %q : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x4x4x2xi8>
  return %r : tensor<1x4x4x2xi8>
}

// Where a residual block changes shape -- every stage transition of every
// ResNet -- the shortcut is a 1x1 projection, so **both** sides of the add are
// accumulators and the tail reads four operands. `matchRequantize` reads one
// accumulator, so neither convolution folds; on ResNet-20 that was 4 of the 21.
// Each gets its own requantization and the add is left for `resadd_i8`.
// CHECK-LABEL: func.func @two_accumulators
// CHECK:         %[[P:.*]] = linalg.conv_2d_nhwc_hwcf
// CHECK:         %[[M:.*]] = linalg.conv_2d_nhwc_hwcf
// CHECK:         %[[A:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[P]], %{{.*}} : tensor<1x8x8x4xi32>, tensor<4xi32>)
// CHECK:           arith.trunci
// CHECK:         %[[B:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[M]], %{{.*}} : tensor<1x8x8x4xi32>, tensor<4xi32>)
// CHECK:           arith.trunci
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<1x8x8x4xi8>, tensor<1x8x8x4xi8>)
// CHECK:           arith.addf

// Both convolutions fold, and the add between them is one more call.
// FOLDED-LABEL: func.func @two_accumulators
// FOLDED:         gemmlir.conv2d_i8
// FOLDED:         gemmlir.conv2d_i8
// FOLDED:         gemmlir.resadd_i8
// FOLDED-NOT:     linalg.conv_2d
func.func @two_accumulators(%in: tensor<1x8x8x4xi8>, %wp: tensor<1x1x4x4xi8>,
                            %wm: tensor<3x3x4x4xi8>, %padded: tensor<1x10x10x4xi8>,
                            %bp: tensor<4xi32>, %bm: tensor<4xi32>)
    -> tensor<1x8x8x4xi8> {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : i32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %sp = arith.constant 5.000000e-04 : f32
  %sm = arith.constant 2.500000e-04 : f32
  %so = arith.constant 2.000000e-02 : f32

  %e = tensor.empty() : tensor<1x8x8x4xi32>
  %fp = linalg.fill ins(%c0 : i32) outs(%e : tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi32>
  %proj = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %wp : tensor<1x8x8x4xi8>, tensor<1x1x4x4xi8>)
    outs(%fp : tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi32>
  %e2 = tensor.empty() : tensor<1x8x8x4xi32>
  %fm = linalg.fill ins(%c0 : i32) outs(%e2 : tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi32>
  %main = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%padded, %wm : tensor<1x10x10x4xi8>, tensor<3x3x4x4xi8>)
    outs(%fm : tensor<1x8x8x4xi32>) -> tensor<1x8x8x4xi32>

  %o = tensor.empty() : tensor<1x8x8x4xi8>
  %r = linalg.generic {indexing_maps = [#nhwc, #nhwc, #chan, #chan, #id],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%proj, %main, %bp, %bm : tensor<1x8x8x4xi32>, tensor<1x8x8x4xi32>, tensor<4xi32>, tensor<4xi32>)
    outs(%o : tensor<1x8x8x4xi8>) {
  ^bb0(%p: i32, %m: i32, %x: i32, %y: i32, %out: i8):
    %ps = arith.addi %p, %x : i32
    %ms = arith.addi %m, %y : i32
    %pf = arith.sitofp %ps : i32 to f32
    %mf = arith.sitofp %ms : i32 to f32
    %pm = arith.mulf %pf, %sp : f32
    %mm = arith.mulf %mf, %sm : f32
    %add = arith.addf %pm, %mm : f32
    %gt = arith.cmpf ugt, %add, %zero : f32
    %relu = arith.select %gt, %add, %zero : f32
    %q = arith.divf %relu, %so : f32
    %rd = math.roundeven %q : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  } -> tensor<1x8x8x4xi8>
  return %r : tensor<1x8x8x4xi8>
}
