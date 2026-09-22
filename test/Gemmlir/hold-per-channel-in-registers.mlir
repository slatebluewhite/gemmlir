// RUN: gemmlir-opt --hold-per-channel-in-registers %s --split-input-file | FileCheck %s

// DenseNet's integer batch norm reloads `M[c]` and `N[c]` on every element,
// because NHWC puts the channel innermost and nothing in this pipeline hoists a
// loop-invariant load. Eight channels at a time become the outermost loop and
// the sixteen coefficients are read once for the whole picture.

#pic = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#idn = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#per = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: func.func @batch_norm
// CHECK:         scf.for %[[G:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// The coefficients come out of the arrays before the picture loops, once.
// CHECK-COUNT-16:  memref.load
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK-COUNT-8:       arith.muli
// CHECK-NOT:           arith.muli
// CHECK-NOT:     linalg.generic
func.func @batch_norm(%in: memref<1x8x8x32xi8>, %m: memref<32xi32>,
                      %n: memref<32xi32>, %out: memref<1x8x8x32xi8>) {
  %c23 = arith.constant 23 : i32
  %c0 = arith.constant 0 : i32
  linalg.generic {indexing_maps = [#pic, #per, #per, #idn],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%in, %m, %n : memref<1x8x8x32xi8>, memref<32xi32>, memref<32xi32>)
    outs(%out : memref<1x8x8x32xi8>) {
  ^bb0(%x: i8, %mm: i32, %nn: i32, %o: i8):
    %e = arith.extsi %x : i8 to i32
    %p = arith.muli %e, %mm : i32
    %a = arith.addi %p, %nn : i32
    %s = arith.shrsi %a, %c23 : i32
    %r = arith.maxsi %s, %c0 : i32
    %t = arith.trunci %r : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// The picture is re-read once per channel group, so it has to fit the L1.
// 16x16x128 is 32 KB and measured **+16.6%** on the board.

#pic = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#idn = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#per = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: func.func @past_the_cache
// CHECK:         linalg.generic
// CHECK-NOT:     scf.for
func.func @past_the_cache(%in: memref<1x16x16x128xi8>, %m: memref<128xi32>,
                          %n: memref<128xi32>, %out: memref<1x16x16x128xi8>) {
  %c23 = arith.constant 23 : i32
  linalg.generic {indexing_maps = [#pic, #per, #per, #idn],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%in, %m, %n : memref<1x16x16x128xi8>, memref<128xi32>, memref<128xi32>)
    outs(%out : memref<1x16x16x128xi8>) {
  ^bb0(%x: i8, %mm: i32, %nn: i32, %o: i8):
    %e = arith.extsi %x : i8 to i32
    %p = arith.muli %e, %mm : i32
    %a = arith.addi %p, %nn : i32
    %s = arith.shrsi %a, %c23 : i32
    %t = arith.trunci %s : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// Sixteen loads need enough pixels to pay for them. Four do not: 2x2x512
// measured +0.7%.

#pic = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#idn = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#per = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: func.func @too_few_pixels
// CHECK:         linalg.generic
// CHECK-NOT:     scf.for
func.func @too_few_pixels(%in: memref<1x2x2x512xi8>, %m: memref<512xi32>,
                          %n: memref<512xi32>, %out: memref<1x2x2x512xi8>) {
  %c23 = arith.constant 23 : i32
  linalg.generic {indexing_maps = [#pic, #per, #per, #idn],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%in, %m, %n : memref<1x2x2x512xi8>, memref<512xi32>, memref<512xi32>)
    outs(%out : memref<1x2x2x512xi8>) {
  ^bb0(%x: i8, %mm: i32, %nn: i32, %o: i8):
    %e = arith.extsi %x : i8 to i32
    %p = arith.muli %e, %mm : i32
    %a = arith.addi %p, %nn : i32
    %s = arith.shrsi %a, %c23 : i32
    %t = arith.trunci %s : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// One runtime f32 a channel is not worth the loop: EfficientNet's gate waits on
// its convert-multiply-convert chain, not on the load, and holding it measured
// -1.5% to +4.4%.

#pic = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#idn = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#per = affine_map<(d0, d1, d2, d3) -> (0, d3, 0, 0)>

// CHECK-LABEL: func.func @one_float_a_channel
// CHECK:         linalg.generic
// CHECK-NOT:     scf.for
func.func @one_float_a_channel(%in: memref<1x8x8x32xi8>, %s: memref<1x32x1x1xf32>,
                               %out: memref<1x8x8x32xi8>) {
  %lo = arith.constant -128 : i32
  linalg.generic {indexing_maps = [#pic, #per, #idn],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%in, %s : memref<1x8x8x32xi8>, memref<1x32x1x1xf32>)
    outs(%out : memref<1x8x8x32xi8>) {
  ^bb0(%x: i8, %f: f32, %o: i8):
    %e = arith.sitofp %x : i8 to f32
    %p = arith.mulf %e, %f : f32
    %r = math.roundeven %p : f32
    %i = arith.fptosi %r : f32 to i32
    %t = arith.trunci %i : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// -----

// A body that reads its own output is an accumulator, which is a different
// operation and a different question.

#pic = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#idn = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#per = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: func.func @accumulates
// CHECK:         linalg.generic
// CHECK-NOT:     scf.for
func.func @accumulates(%in: memref<1x8x8x32xi8>, %m: memref<32xi32>,
                       %n: memref<32xi32>, %out: memref<1x8x8x32xi8>) {
  %c23 = arith.constant 23 : i32
  linalg.generic {indexing_maps = [#pic, #per, #per, #idn],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%in, %m, %n : memref<1x8x8x32xi8>, memref<32xi32>, memref<32xi32>)
    outs(%out : memref<1x8x8x32xi8>) {
  ^bb0(%x: i8, %mm: i32, %nn: i32, %o: i8):
    %e = arith.extsi %x : i8 to i32
    %p = arith.muli %e, %mm : i32
    %a = arith.addi %p, %nn : i32
    %s = arith.shrsi %a, %c23 : i32
    %t = arith.trunci %s : i32 to i8
    %k = arith.maxsi %t, %o : i8
    linalg.yield %k : i8
  }
  return
}
