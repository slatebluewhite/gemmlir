// A residual block's activation has two consumers: the next convolution, which
// wants it quantized, and the shortcut, which wants it in f32. Quantizing it
// where each consumer reads it leaves the producing convolution's tail in f32,
// so the tail never ends in a requantization and the convolution does not fold.

// RUN: gemmlir-opt --share-branch-quantization %s | FileCheck %s

!qi8 = !quant.uniform<i8:f32, 2.000000e-02>
!qi32 = !quant.uniform<i32:f32, 4.000000e-04>

// The quantization moves to the branch, and it moves into the *convolution's*
// layout: the transpose a frontend leaves between the tail and the convolution
// is hoisted in front of the quantization, where the absorb-and-fuse machinery
// downstream can pull it into the tail. The pad follows on i8, with its amounts
// permuted along with it -- NCHW's low[0, 0, 1, 1] is NHWC's low[0, 1, 1, 0].
// The shortcut reads the same i8 back.
// CHECK-LABEL: func.func @residual
// CHECK:         %[[TAIL:.*]] = linalg.generic
// CHECK:           arith.select
// CHECK:         %[[T:.*]] = linalg.transpose ins(%[[TAIL]]
// CHECK-SAME:      permutation = [0, 2, 3, 1]
// CHECK:         %[[Q:.*]] = quant.qcast %[[T]]
// CHECK:         %[[I8:.*]] = quant.scast %[[Q]]
// CHECK-SAME:      to tensor<1x4x4x2xi8>
// CHECK:         tensor.pad %[[I8]] low[0, 1, 1, 0] high[0, 1, 1, 0]
// CHECK:         %[[W:.*]] = arith.sitofp %[[I8]]
// CHECK:         %[[D:.*]] = arith.mulf %[[W]]
// CHECK:         linalg.transpose ins(%[[D]]
// CHECK-SAME:      permutation = [0, 3, 1, 2]
func.func @residual(%acc: tensor<1x4x4x2xi32>) -> (tensor<1x2x4x4xf32>, tensor<1x6x6x2xi8>) {
  %z = arith.constant 0.0 : f32
  %q = quant.scast %acc : tensor<1x4x4x2xi32> to tensor<1x4x4x2x!qi32>
  %d = quant.dcast %q : tensor<1x4x4x2x!qi32> to tensor<1x4x4x2xf32>
  %e = tensor.empty() : tensor<1x2x4x4xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(0,d2,d3,d1)>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%d : tensor<1x4x4x2xf32>) outs(%e : tensor<1x2x4x4xf32>) {
  ^bb0(%in: f32, %o: f32):
    %c = arith.cmpf ugt, %in, %z : f32
    %s = arith.select %c, %in, %z : f32
    linalg.yield %s : f32
  } -> tensor<1x2x4x4xf32>
  %p = tensor.pad %relu low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c2: index, %d2: index):
    tensor.yield %z : f32
  } : tensor<1x2x4x4xf32> to tensor<1x2x6x6xf32>
  %e2 = tensor.empty() : tensor<1x6x6x2xf32>
  %t = linalg.transpose ins(%p : tensor<1x2x6x6xf32>) outs(%e2 : tensor<1x6x6x2xf32>) permutation = [0, 2, 3, 1]
  %qc = quant.qcast %t : tensor<1x6x6x2xf32> to tensor<1x6x6x2x!qi8>
  %s8 = quant.scast %qc : tensor<1x6x6x2x!qi8> to tensor<1x6x6x2xi8>
  return %relu, %s8 : tensor<1x2x4x4xf32>, tensor<1x6x6x2xi8>
}

// The dequantization is written out as arithmetic rather than as a
// `quant.dcast`. The quant dialect folds `dcast(qcast(x))` straight back to
// `x` -- it takes a quantization to be exact -- which undoes the rewrite as
// fast as it is made, and the greedy driver then never terminates.
// CHECK-LABEL: func.func @no_dcast
// CHECK:         quant.qcast
// CHECK:         %[[I8:.*]] = quant.scast
// CHECK-NOT:     quant.dcast
// CHECK:         arith.sitofp %[[I8]]
// CHECK:         arith.mulf
func.func @no_dcast(%acc: tensor<8xi32>) -> (tensor<8xf32>, tensor<8xi8>) {
  %q = quant.scast %acc : tensor<8xi32> to tensor<8x!qi32>
  %d = quant.dcast %q : tensor<8x!qi32> to tensor<8xf32>
  %qc = quant.qcast %d : tensor<8xf32> to tensor<8x!qi8>
  %s8 = quant.scast %qc : tensor<8x!qi8> to tensor<8xi8>
  return %d, %s8 : tensor<8xf32>, tensor<8xi8>
}

// A value with one consumer is not a branch: quantizing it where it is read is
// already the right place, and there is nothing to share.
// CHECK-LABEL: func.func @single_consumer
// CHECK-NOT:     arith.sitofp
// CHECK:         return
func.func @single_consumer(%acc: tensor<8xi32>) -> tensor<8xi8> {
  %q = quant.scast %acc : tensor<8xi32> to tensor<8x!qi32>
  %d = quant.dcast %q : tensor<8x!qi32> to tensor<8xf32>
  %qc = quant.qcast %d : tensor<8xf32> to tensor<8x!qi8>
  %s8 = quant.scast %qc : tensor<8x!qi8> to tensor<8xi8>
  return %s8 : tensor<8xi8>
}

// A branch one of whose consumers is the function's own result is left alone.
// Every other consumer would get the dequantization of an i8 instead of the f32
// it had, which is free when that consumer was going to quantize it anyway --
// and this one is not a consumer at all, it is the answer.
// CHECK-LABEL: func.func @not_a_contraction_tail
// CHECK-NOT:     arith.sitofp
// CHECK:         return
func.func @not_a_contraction_tail(%x: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xi8>) {
  %two = arith.constant dense<2.0> : tensor<8xf32>
  %y = arith.mulf %x, %two : tensor<8xf32>
  %qc = quant.qcast %y : tensor<8xf32> to tensor<8x!qi8>
  %s8 = quant.scast %qc : tensor<8x!qi8> to tensor<8xi8>
  return %y, %s8 : tensor<8xf32>, tensor<8xi8>
}

// A padding value other than zero is not what a symmetric quantization maps to
// itself, nor what the accelerator's `padding` produces, so the pad is not
// moved into i8 -- the walk stops there and the branch above it is not found.
// CHECK-LABEL: func.func @nonzero_padding
// CHECK-NOT:     arith.sitofp
// CHECK:         return
func.func @nonzero_padding(%acc: tensor<1x4xi32>) -> (tensor<1x4xf32>, tensor<1x6xi8>) {
  %one = arith.constant 1.0 : f32
  %q = quant.scast %acc : tensor<1x4xi32> to tensor<1x4x!qi32>
  %d = quant.dcast %q : tensor<1x4x!qi32> to tensor<1x4xf32>
  %p = tensor.pad %d low[0, 1] high[0, 1] {
  ^bb0(%a: index, %b: index):
    tensor.yield %one : f32
  } : tensor<1x4xf32> to tensor<1x6xf32>
  %qc = quant.qcast %p : tensor<1x6xf32> to tensor<1x6x!qi8>
  %s8 = quant.scast %qc : tensor<1x6x!qi8> to tensor<1x6xi8>
  return %d, %s8 : tensor<1x4xf32>, tensor<1x6xi8>
}

// Where the channels are split -- ShuffleNet's unit passes half of them through
// untouched -- the branch reaches its convolution through a `tensor.extract_slice`
// as well. The slice is rebuilt on i8 like the pad is, so the tail ends in a
// requantization and the convolution above it folds.
// CHECK-LABEL: func.func @split_channels
// CHECK:         %[[Q:.*]] = quant.qcast %{{.*}} : tensor<1x4x4x16xf32>
// CHECK:         %[[I8:.*]] = quant.scast %[[Q]]
// CHECK-SAME:      to tensor<1x4x4x16xi8>
// CHECK:         tensor.extract_slice %[[I8]][0, 0, 0, 8] [1, 4, 4, 8] [1, 1, 1, 1]
// CHECK:         arith.sitofp %[[I8]]
func.func @split_channels(%acc: tensor<1x4x4x16xi32>)
    -> (tensor<1x4x4x16xf32>, tensor<1x4x4x8xi8>) {
  %q = quant.scast %acc : tensor<1x4x4x16xi32> to tensor<1x4x4x16x!qi32>
  %d = quant.dcast %q : tensor<1x4x4x16x!qi32> to tensor<1x4x4x16xf32>
  %sl = tensor.extract_slice %d[0, 0, 0, 8] [1, 4, 4, 8] [1, 1, 1, 1]
        : tensor<1x4x4x16xf32> to tensor<1x4x4x8xf32>
  %qc = quant.qcast %sl : tensor<1x4x4x8xf32> to tensor<1x4x4x8x!qi8>
  %s8 = quant.scast %qc : tensor<1x4x4x8x!qi8> to tensor<1x4x4x8xi8>
  return %d, %s8 : tensor<1x4x4x16xf32>, tensor<1x4x4x8xi8>
}

// A reshape on the way is rebuilt too. `isLayoutOnly` lists reshapes on purpose
// -- a grouped convolution arrives as one -- but the replay below only knew how
// to put back slices and pads, so a reshape was dropped and the storage cast
// kept the root's shape while the cast it replaced had the reshaped one:
// `'quant.scast' op failed to verify`. A squeeze-excite block's pool is
// `1x4x4x16 -> 16x16`, and it is what stopped EfficientNet compiling at all.
// CHECK-LABEL: func.func @collapse_on_the_way
// CHECK:         %[[Q:.*]] = quant.qcast %{{.*}} : tensor<1x4x4x16xf32>
// CHECK:         %[[I8:.*]] = quant.scast %[[Q]]
// CHECK-SAME:      to tensor<1x4x4x16xi8>
// CHECK:         tensor.collapse_shape %[[I8]] {{\[}}[0, 1, 2], [3]]
// CHECK-SAME:      tensor<1x4x4x16xi8> into tensor<16x16xi8>
// CHECK:         arith.sitofp %[[I8]]
func.func @collapse_on_the_way(%acc: tensor<1x4x4x16xi32>)
    -> (tensor<1x4x4x16xf32>, tensor<16x16xi8>) {
  %q = quant.scast %acc : tensor<1x4x4x16xi32> to tensor<1x4x4x16x!qi32>
  %d = quant.dcast %q : tensor<1x4x4x16x!qi32> to tensor<1x4x4x16xf32>
  %c = tensor.collapse_shape %d [[0, 1, 2], [3]]
     : tensor<1x4x4x16xf32> into tensor<16x16xf32>
  %qc = quant.qcast %c : tensor<16x16xf32> to tensor<16x16x!qi8>
  %s8 = quant.scast %qc : tensor<16x16x!qi8> to tensor<16x16xi8>
  return %d, %s8 : tensor<1x4x4x16xf32>, tensor<16x16xi8>
}

// A transpose *after* a reshape is where the per-dimension bookkeeping stops
// applying -- the two have different ranks -- so that chain is left alone
// rather than rebuilt wrong.
// CHECK-LABEL: func.func @transpose_after_a_reshape
// CHECK-NOT:     tensor.collapse_shape %{{.*}}i8
// CHECK:         quant.qcast %{{.*}} : tensor<16x16xf32>
func.func @transpose_after_a_reshape(%acc: tensor<1x4x4x16xi32>)
    -> (tensor<1x4x4x16xf32>, tensor<16x16xi8>) {
  %q = quant.scast %acc : tensor<1x4x4x16xi32> to tensor<1x4x4x16x!qi32>
  %d = quant.dcast %q : tensor<1x4x4x16x!qi32> to tensor<1x4x4x16xf32>
  %c = tensor.collapse_shape %d [[0, 1, 2], [3]]
     : tensor<1x4x4x16xf32> into tensor<16x16xf32>
  %e = tensor.empty() : tensor<16x16xf32>
  %t = linalg.transpose ins(%c : tensor<16x16xf32>) outs(%e : tensor<16x16xf32>)
       permutation = [1, 0]
  %qc = quant.qcast %t : tensor<16x16xf32> to tensor<16x16x!qi8>
  %s8 = quant.scast %qc : tensor<16x16x!qi8> to tensor<16x16xi8>
  return %d, %s8 : tensor<1x4x4x16xf32>, tensor<16x16xi8>
}

// The search for the contraction above a branch has a budget, and the budget
// has to be longer than the chains that occur. ResNet-18's max-pool is **seven**
// steps from its accumulator -- residual add, relayout, tail, relayout, pad,
// pool -- and at six it was refused. Both of that model's remaining scalar
// convolutions came from this one refusal: the pool's two consumers quantized
// it separately, so the stem's tail stayed f32 and the block below ended in a
// requantization with three inputs.
// CHECK-LABEL: func.func @seven_steps_up
// CHECK:         %[[P:.*]] = linalg.pooling_nhwc_max
// CHECK:         %[[Q:.*]] = quant.qcast %[[P]]
// CHECK:         %[[I8:.*]] = quant.scast %[[Q]]
// CHECK:         arith.sitofp %[[I8]]
func.func @seven_steps_up(%acc: tensor<1x8x8x16xi32>, %other: tensor<1x8x8x16xf32>)
    -> (tensor<1x4x4x16xf32>, tensor<1x4x4x16xi8>) {
  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %q = quant.scast %acc : tensor<1x8x8x16xi32> to tensor<1x8x8x16x!qi32>
  %d = quant.dcast %q : tensor<1x8x8x16x!qi32> to tensor<1x8x8x16xf32>
  // 1: the residual add
  %sum = arith.addf %d, %other : tensor<1x8x8x16xf32>
  // 2: into NCHW
  %e1 = tensor.empty() : tensor<1x16x8x8xf32>
  %t1 = linalg.transpose ins(%sum : tensor<1x8x8x16xf32>) outs(%e1 : tensor<1x16x8x8xf32>) permutation = [0, 3, 1, 2]
  // 3: the tail
  %e2 = tensor.empty() : tensor<1x16x8x8xf32>
  %tail = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                                           affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
                          iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%t1 : tensor<1x16x8x8xf32>) outs(%e2 : tensor<1x16x8x8xf32>) {
  ^bb0(%in: f32, %o: f32):
    %c = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %c, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x16x8x8xf32>
  // 4: back to NHWC
  %e3 = tensor.empty() : tensor<1x8x8x16xf32>
  %t2 = linalg.transpose ins(%tail : tensor<1x16x8x8xf32>) outs(%e3 : tensor<1x8x8x16xf32>) permutation = [0, 2, 3, 1]
  // 5: the pool's own padding
  %pd = tensor.pad %t2 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %e: index):
    tensor.yield %zero : f32
  } : tensor<1x8x8x16xf32> to tensor<1x10x10x16xf32>
  // 6: the pool -- and its result is the branch
  %w = tensor.empty() : tensor<3x3xf32>
  %pe = tensor.empty() : tensor<1x4x4x16xf32>
  %pi = linalg.fill ins(%ninf : f32) outs(%pe : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
  %pool = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
     ins(%pd, %w : tensor<1x10x10x16xf32>, tensor<3x3xf32>) outs(%pi : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
  %qc = quant.qcast %pool : tensor<1x4x4x16xf32> to tensor<1x4x4x16x!qi8>
  %s8 = quant.scast %qc : tensor<1x4x4x16x!qi8> to tensor<1x4x4x16xi8>
  return %pool, %s8 : tensor<1x4x4x16xf32>, tensor<1x4x4x16xi8>
}

// A branch with no contraction above it at all, where **every** consumer is a
// quantization asking for the same thing: shared. Quantizing here costs the
// consumer that wanted it nothing and turns the other consumer's read of an f32
// tensor into a read of an i8 one. EfficientNet's sixteen squeeze-excitation
// blocks are this shape -- **551.4 -> 400.8 ms** on EfficientNet and
// **278.5 -> 150.6** on RegNet.
// CHECK-LABEL: func.func @quantized_anyway_is_shared
// CHECK:         %[[Y:.*]] = arith.mulf
// CHECK:         %[[Q:.*]] = quant.qcast %[[Y]]
// CHECK:         %[[S:.*]] = quant.scast %[[Q]]
// CHECK:         %[[W:.*]] = arith.sitofp %[[S]]
// CHECK:         arith.mulf %[[W]]
func.func @quantized_anyway_is_shared(%x: tensor<8xf32>, %g: tensor<8xf32>)
    -> (tensor<8xi8>, tensor<8xi8>) {
  %two = arith.constant dense<2.000000e+00> : tensor<8xf32>
  %y = arith.mulf %x, %two : tensor<8xf32>
  %qc = quant.qcast %y : tensor<8xf32> to tensor<8x!qi8>
  %s8 = quant.scast %qc : tensor<8x!qi8> to tensor<8xi8>
  %z = arith.mulf %y, %g : tensor<8xf32>
  %zq = quant.qcast %z : tensor<8xf32> to tensor<8x!qi8>
  %z8 = quant.scast %zq : tensor<8x!qi8> to tensor<8xi8>
  return %z8, %s8 : tensor<8xi8>, tensor<8xi8>
}

!qi8b = !quant.uniform<i8:f32, 1.000000e-02>

// Two branches of one value that were calibrated separately, so they ask for
// two different scales. Hoisting either one over both is not free: the branch
// that needed 0.02 saturates at 0.01, and that is a rounding the other branch's
// quantization was never going to absorb.
//
// This is the LSTM, whose input is sixteen timesteps sliced out of one tensor
// with scales from 0.0128 to 0.0248. Sharing the last one over all sixteen took
// the model from 0.0066 to **0.0284** relative L2 for 2% slower. Three other
// conditions were tried before this one and none of them separated it from the
// squeeze-excitation case above; see ShareBranchQuantizationPass.cpp.
// CHECK-LABEL: func.func @branches_disagree_on_the_scale
// CHECK-NOT:     arith.sitofp
// CHECK:         return
func.func @branches_disagree_on_the_scale(%x: tensor<2x8xf32>)
    -> (tensor<8xi8>, tensor<8xi8>) {
  %a = tensor.extract_slice %x[0, 0] [1, 8] [1, 1] : tensor<2x8xf32> to tensor<8xf32>
  %qa = quant.qcast %a : tensor<8xf32> to tensor<8x!qi8>
  %a8 = quant.scast %qa : tensor<8x!qi8> to tensor<8xi8>
  %b = tensor.extract_slice %x[1, 0] [1, 8] [1, 1] : tensor<2x8xf32> to tensor<8xf32>
  %qb = quant.qcast %b : tensor<8xf32> to tensor<8x!qi8b>
  %b8 = quant.scast %qb : tensor<8x!qi8b> to tensor<8xi8>
  return %a8, %b8 : tensor<8xi8>, tensor<8xi8>
}

!qi8c = !quant.uniform<i8:f32, 2.000000e-02>
!qi32c = !quant.uniform<i32:f32, 4.000000e-04>

// A relayout written as a `linalg.generic` rather than a `linalg.transpose`.
//
// torch-mlir writes the same thing either way: a body that yields its input
// unchanged, with the permutation on the **write** map --
// `out[d0, d2, d1, d3] = in[d0, d1, d2, d3]`. A transformer's attention reaches
// its QKV split through three of those, and a walk that only knows the named
// operation stops one step below the split.
//
// This does not reach ViT's own QKV join -- there the three branches quantize
// below the split and no `quant.qcast` sits on the join for this pass to anchor
// on, so the walk never gets there. It is the shape that would have blocked it
// if one did, and every model in the set compiles to a byte-identical object
// either way. Kept because the walk is meant to be about layout, not about
// which spelling of a relayout the frontend chose.
// The relayout is hoisted in front of the quantization and rebuilt as the named
// operation, which is what the rest of the pipeline matches on; the shortcut
// then reads the dequantization of the shared i8.
// CHECK-LABEL: func.func @a_relayout_written_as_a_generic
// CHECK:         %[[T:.*]] = linalg.transpose
// CHECK-SAME:      permutation = [0, 2, 1, 3]
// CHECK:         %[[Q:.*]] = quant.qcast %[[T]]
// CHECK-SAME:      to tensor<1x4x2x8x!quant.uniform<i8:f32, 2.000000e-02>>
// CHECK:         %[[S:.*]] = quant.scast %[[Q]]
// CHECK:         arith.sitofp %[[S]]
// CHECK-NOT:     linalg.generic
func.func @a_relayout_written_as_a_generic(%acc: tensor<1x2x4x8xi32>)
    -> (tensor<1x2x4x8xf32>, tensor<1x4x2x8xi8>) {
  %q = quant.scast %acc : tensor<1x2x4x8xi32> to tensor<1x2x4x8x!qi32c>
  %d = quant.dcast %q : tensor<1x2x4x8x!qi32c> to tensor<1x2x4x8xf32>
  %e = tensor.empty() : tensor<1x4x2x8xf32>
  %t = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%d : tensor<1x2x4x8xf32>) outs(%e : tensor<1x4x2x8xf32>) {
  ^bb0(%v: f32, %out: f32):
    linalg.yield %v : f32
  } -> tensor<1x4x2x8xf32>
  %qc = quant.qcast %t : tensor<1x4x2x8xf32> to tensor<1x4x2x8x!qi8c>
  %s8 = quant.scast %qc : tensor<1x4x2x8x!qi8c> to tensor<1x4x2x8xi8>
  return %d, %s8 : tensor<1x2x4x8xf32>, tensor<1x4x2x8xi8>
}
