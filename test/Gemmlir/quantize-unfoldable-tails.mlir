// RUN: gemmlir-opt --split-input-file --quantize-unfoldable-tails %s | FileCheck %s

// `conv2d_i8` and `depthwise_conv2d_i8` write i8, so a layer's tail has to be
// `saturate(scale * accumulator + bias)` with at most a relu. EfficientNet's
// activation is SiLU, `x * sigmoid(x)`, which is none of those.
//
// This used to be confined to a *depthwise* convolution, on the reasoning that a
// dense one can always be packed as a matmul and a matmul may leave its i32
// accumulator for the core to finish. The board says otherwise: letting a dense
// convolution take the cut too is EfficientNet **1114 -> 692 ms**, because
// fourteen of its convolutions stop being packed matmuls and become
// `conv2d_i8`. A cut is not only for a layer with nowhere else to go; it is for
// whichever layer ends up faster with one.
//
// Quantizing the layer's own output splits the tail into the part the call can
// do and the part the core does afterwards. The scale is the layer's own output
// range, which the calibration measures separately because an activation sits
// between it and the next layer's input.

!qi32 = !quant.uniform<i32:f32, 4.000000e-05>

#chan = affine_map<(d0, d1, d2, d3) -> (d3)>
#id4  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @silu_gets_a_requantization
// The bias stays on the accelerator's side of the cut...
// CHECK:         %[[B:[a-z0-9_]+]] = arith.addf %{{.*}} : tensor<1x8x8x16xf32>
// ...and the tail is quantized there.
// CHECK:         %[[Q:[a-z0-9_]+]] = quant.qcast %[[B]]
// CHECK-SAME:      to tensor<1x8x8x16x!quant.uniform<i8:f32, 1.000000e-02>>
// CHECK:         %[[S:[a-z0-9_]+]] = quant.scast %[[Q]]
// CHECK-SAME:      to tensor<1x8x8x16xi8>
// Written as arithmetic: the quant dialect folds `dcast(qcast(x))` back to `x`.
// CHECK:         %[[W:[a-z0-9_]+]] = arith.sitofp %[[S]]
// CHECK:         %[[D:[a-z0-9_]+]] = arith.mulf %[[W]], %{{.*}}
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[D]]
func.func @silu_gets_a_requantization(%acc: tensor<1x8x8x16xi8>, %flt: tensor<1x1x16xi8>,
                                      %bias: tensor<16xf32>) -> tensor<1x8x8x16xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<1x8x8x16xi32>
  %be = tensor.empty() : tensor<1x8x8x16xf32>
  %f = linalg.generic {indexing_maps = [#chan, #id4],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%bias : tensor<16xf32>) outs(%be : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<1x8x8x16xf32>
  %c = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
                                          strides = dense<1> : tensor<2xi64>,
                                          gemmlir.output_scale = 1.000000e-02 : f64}
      ins(%acc, %flt : tensor<1x8x8x16xi8>, tensor<1x1x16xi8>)
      outs(%e : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  %q = quant.scast %c : tensor<1x8x8x16xi32> to tensor<1x8x8x16x!qi32>
  %d = quant.dcast %q : tensor<1x8x8x16x!qi32> to tensor<1x8x8x16xf32>
  %b = arith.addf %d, %f : tensor<1x8x8x16xf32>
  // SiLU: the value is read twice, by the sigmoid and by the multiply.
  %se = tensor.empty() : tensor<1x8x8x16xf32>
  %sig = linalg.generic {indexing_maps = [#id4, #id4],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%b : tensor<1x8x8x16xf32>) outs(%se : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %n = arith.negf %in : f32
    %x = math.exp %n : f32
    %p = arith.addf %x, %one : f32
    %r = arith.divf %one, %p : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x16xf32>
  %out = arith.mulf %sig, %b : tensor<1x8x8x16xf32>
  return %out : tensor<1x8x8x16xf32>
}

// -----

// A tail the pipeline *can* do is left alone: a relu and a quantization is
// exactly what the mvout does, and cutting it in two would only lose a bit.

!qi32 = !quant.uniform<i32:f32, 4.000000e-05>
!qi8 = !quant.uniform<i8:f32, 2.000000e-02>

#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @a_relu_tail_is_left_alone
// CHECK-NOT:     quant.qcast %{{.*}} to tensor<1x8x8x16x!quant.uniform<i8:f32, 1.000000e-02>>
func.func @a_relu_tail_is_left_alone(%acc: tensor<1x8x8x16xi8>, %flt: tensor<1x1x16xi8>) -> tensor<1x8x8x16xi8> {
  %zero = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<1x8x8x16xi32>
  %c = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
                                          strides = dense<1> : tensor<2xi64>,
                                          gemmlir.output_scale = 1.000000e-02 : f64}
      ins(%acc, %flt : tensor<1x8x8x16xi8>, tensor<1x1x16xi8>)
      outs(%e : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  %q = quant.scast %c : tensor<1x8x8x16xi32> to tensor<1x8x8x16x!qi32>
  %d = quant.dcast %q : tensor<1x8x8x16x!qi32> to tensor<1x8x8x16xf32>
  %z = tensor.empty() : tensor<1x8x8x16xf32>
  %zf = linalg.fill ins(%zero : f32) outs(%z : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %r = arith.maximumf %d, %zf : tensor<1x8x8x16xf32>
  %qc = quant.qcast %r : tensor<1x8x8x16xf32> to tensor<1x8x8x16x!qi8>
  %s8 = quant.scast %qc : tensor<1x8x8x16x!qi8> to tensor<1x8x8x16xi8>
  return %s8 : tensor<1x8x8x16xi8>
}

// -----

// Without a measured output range there is nothing to quantize at, and the
// layer keeps its loop rather than guessing a scale.

!qi32 = !quant.uniform<i32:f32, 4.000000e-05>

#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @no_scale_no_cut
// CHECK-NOT:     quant.qcast
func.func @no_scale_no_cut(%acc: tensor<1x8x8x16xi8>, %flt: tensor<1x1x16xi8>) -> tensor<1x8x8x16xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<1x8x8x16xi32>
  %c = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
                                          strides = dense<1> : tensor<2xi64>}
      ins(%acc, %flt : tensor<1x8x8x16xi8>, tensor<1x1x16xi8>)
      outs(%e : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  %q = quant.scast %c : tensor<1x8x8x16xi32> to tensor<1x8x8x16x!qi32>
  %d = quant.dcast %q : tensor<1x8x8x16x!qi32> to tensor<1x8x8x16xf32>
  %se = tensor.empty() : tensor<1x8x8x16xf32>
  %sig = linalg.generic {indexing_maps = [#id4, #id4],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%d : tensor<1x8x8x16xf32>) outs(%se : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %n = arith.negf %in : f32
    %x = math.exp %n : f32
    linalg.yield %x : f32
  } -> tensor<1x8x8x16xf32>
  return %sig : tensor<1x8x8x16xf32>
}

// -----

// A **reduction** stops a tail just as surely as a transcendental does: the
// accelerator writes its own output and a layer norm's mean does not have that
// shape. ConvNeXt puts one under every one of its depthwise convolutions, and
// they were the eighteen contractions it left on the core.
//
// The bias above it arrives as a *transposed* broadcast, because the layout
// rewrite relays it out like everything else -- reading only the transpose made
// each of them look like a residual add, which is left alone on purpose.

!qi32 = !quant.uniform<i32:f32, 4.000000e-05>

#chan = affine_map<(d0, d1, d2, d3) -> (d3)>
#id4  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#drop = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @a_layer_norm_stops_the_tail
// CHECK:         %[[B:[a-z0-9_]+]] = arith.addf
// CHECK:         %[[Q:[a-z0-9_]+]] = quant.qcast %[[B]]
// CHECK-SAME:      to tensor<1x8x8x16x!quant.uniform<i8:f32, 1.000000e-02>>
// CHECK:         quant.scast %[[Q]]
// CHECK:         arith.sitofp
func.func @a_layer_norm_stops_the_tail(%acc: tensor<1x8x8x16xi8>, %flt: tensor<1x1x16xi8>,
                                       %bias: tensor<16xf32>) -> tensor<1x8x8xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<1x8x8x16xi32>
  %be = tensor.empty() : tensor<1x8x8x16xf32>
  %bt = tensor.empty() : tensor<1x8x8x16xf32>
  %c = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
                                          strides = dense<1> : tensor<2xi64>,
                                          gemmlir.output_scale = 1.000000e-02 : f64}
      ins(%acc, %flt : tensor<1x8x8x16xi8>, tensor<1x1x16xi8>)
      outs(%e : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  %q = quant.scast %c : tensor<1x8x8x16xi32> to tensor<1x8x8x16x!qi32>
  %d = quant.dcast %q : tensor<1x8x8x16x!qi32> to tensor<1x8x8x16xf32>
  %bc = linalg.generic {indexing_maps = [#chan, #id4],
                        iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%bias : tensor<16xf32>) outs(%be : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<1x8x8x16xf32>
  // the layout rewrite's relayout of the bias
  %bp = linalg.transpose ins(%bc : tensor<1x8x8x16xf32>) outs(%bt : tensor<1x8x8x16xf32>)
        permutation = [0, 1, 2, 3]
  %b = arith.addf %d, %bp : tensor<1x8x8x16xf32>
  // the mean: a reduction, read alongside the value itself
  %me = tensor.empty() : tensor<1x8x8xf32>
  %mi = linalg.fill ins(%zero : f32) outs(%me : tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
  %mean = linalg.generic {indexing_maps = [#id4, #drop],
                          iterator_types = ["parallel","parallel","parallel","reduction"]}
      ins(%b : tensor<1x8x8x16xf32>) outs(%mi : tensor<1x8x8xf32>) {
  ^bb0(%in: f32, %o: f32):
    %s = arith.addf %in, %o : f32
    linalg.yield %s : f32
  } -> tensor<1x8x8xf32>
  return %mean : tensor<1x8x8xf32>
}

// -----

// The same cut under a **dense** convolution. Nothing about the tail changes --
// a sigmoid is no more foldable here than it was above -- and what the cut buys
// is that the layer can now be a `conv2d_i8` instead of an img2col pack and a
// `matmul_i8` that leaves its accumulator behind.

!qi32 = !quant.uniform<i32:f32, 4.000000e-05>

#chan = affine_map<(d0, d1, d2, d3) -> (d3)>
#id4  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @a_dense_convolution_takes_the_cut_too
// CHECK:         %[[B:[a-z0-9_]+]] = arith.addf
// CHECK:         %[[Q:[a-z0-9_]+]] = quant.qcast %[[B]]
// CHECK-SAME:      to tensor<1x6x6x16x!quant.uniform<i8:f32, 1.000000e-02>>
// CHECK:         quant.scast %[[Q]]
// CHECK:         arith.sitofp
func.func @a_dense_convolution_takes_the_cut_too(%acc: tensor<1x8x8x16xi8>,
                                                 %flt: tensor<3x3x16x16xi8>,
                                                 %bias: tensor<16xf32>)
    -> tensor<1x6x6x16xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<1x6x6x16xi32>
  %be = tensor.empty() : tensor<1x6x6x16xf32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>,
                                 gemmlir.output_scale = 1.000000e-02 : f64}
      ins(%acc, %flt : tensor<1x8x8x16xi8>, tensor<3x3x16x16xi8>)
      outs(%e : tensor<1x6x6x16xi32>) -> tensor<1x6x6x16xi32>
  %q = quant.scast %c : tensor<1x6x6x16xi32> to tensor<1x6x6x16x!qi32>
  %d = quant.dcast %q : tensor<1x6x6x16x!qi32> to tensor<1x6x6x16xf32>
  %bc = linalg.generic {indexing_maps = [#chan, #id4],
                        iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%bias : tensor<16xf32>) outs(%be : tensor<1x6x6x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<1x6x6x16xf32>
  %b = arith.addf %d, %bc : tensor<1x6x6x16xf32>
  // SiLU: x * sigmoid(x), which no mvout tail can end in
  %so = tensor.empty() : tensor<1x6x6x16xf32>
  %sig = linalg.generic {indexing_maps = [#id4, #id4],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%b : tensor<1x6x6x16xf32>) outs(%so : tensor<1x6x6x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %n = arith.negf %in : f32
    %ex = math.exp %n : f32
    %de = arith.addf %ex, %one : f32
    %si = arith.divf %in, %de : f32
    linalg.yield %si : f32
  } -> tensor<1x6x6x16xf32>
  return %sig : tensor<1x6x6x16xf32>
}

// -----

// The same tail, written the way a **transformer** writes it, which is two
// spellings away from the one above.
//
// A convolution network's bias is a tensor-level `arith.addf` against a
// broadcast. A ViT's is one `linalg.generic` reading the accumulator and the
// bias together -- three operands and no `arith` op at this level -- so the walk
// stopped on it and called it the blocker, and every one of the twelve GELUs
// stayed on the other side of that stop.
//
// And the batch of one is written as the **constant 0** rather than as a loop:
// `(d0, d1, d2) -> (0, d1, d2)`, which on an extent of one is the identity and
// which `isIdentity()` calls no. Asking that question answered no to all twelve.
//
// Measured: the twelve GELUs are 156,672 `erff` calls an inference, and cutting
// here lets `--table-for-i8-elementwise` have them. **955.13 -> 468.55 ms**, at
// 0.0553 relative L2 against 0.0452 -- the cut is one more quantization and on a
// transformer it is not free.

!qi32 = !quant.uniform<i32:f32, 1.500000e-05>

#unit = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#chan = affine_map<(d0, d1, d2) -> (d2)>
#id3  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#id2  = affine_map<(d0, d1) -> (d0, d1)>

// The cut lands after the bias, which is where the calibration measured, and
// before the GELU.
// CHECK-LABEL: func.func @a_generic_bias_and_a_unit_axis
// CHECK:         %[[B:[a-z0-9_]+]] = linalg.generic
// CHECK:           arith.addf
// CHECK:         %[[Q:[a-z0-9_]+]] = quant.qcast %[[B]]
// CHECK-SAME:      to tensor<1x17x768x!quant.uniform<i8:f32, 1.970000e-02>>
// CHECK:         quant.scast %[[Q]]
// CHECK:         arith.sitofp
// CHECK:         math.erf
func.func @a_generic_bias_and_a_unit_axis(%a: tensor<17x192xi8>, %w: tensor<192x768xi8>,
                                          %bias: tensor<768xf32>) -> tensor<17x768xf32> {
  %z = arith.constant 0 : i32
  %half = arith.constant 5.000000e-01 : f32
  %one = arith.constant 1.000000e+00 : f32
  %r2 = arith.constant 1.414213562 : f32
  %e = tensor.empty() : tensor<17x768xi32>
  %init = linalg.fill ins(%z : i32) outs(%e : tensor<17x768xi32>) -> tensor<17x768xi32>
  %m = linalg.matmul {gemmlir.output_scale = 1.970000e-02 : f64}
      ins(%a, %w : tensor<17x192xi8>, tensor<192x768xi8>)
      outs(%init : tensor<17x768xi32>) -> tensor<17x768xi32>
  %s = quant.scast %m : tensor<17x768xi32> to tensor<17x768x!qi32>
  %d = quant.dcast %s : tensor<17x768x!qi32> to tensor<17x768xf32>
  %ex = tensor.expand_shape %d [[0, 1], [2]] output_shape [1, 17, 768]
      : tensor<17x768xf32> into tensor<1x17x768xf32>
  %be = tensor.empty() : tensor<1x17x768xf32>
  // the bias, as a generic, with the batch axis read as a constant 0
  %b = linalg.generic {indexing_maps = [#unit, #chan, #id3],
                       iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ex, %bias : tensor<1x17x768xf32>, tensor<768xf32>) outs(%be : tensor<1x17x768xf32>) {
  ^bb0(%in: f32, %bb: f32, %o: f32):
    %t = arith.addf %in, %bb : f32
    linalg.yield %t : f32
  } -> tensor<1x17x768xf32>
  %co = tensor.collapse_shape %b [[0, 1], [2]] : tensor<1x17x768xf32> into tensor<17x768xf32>
  %ge = tensor.empty() : tensor<17x768xf32>
  // GELU: 0.5x(1 + erf(x/sqrt2)), which no mvout tail can end in
  %g = linalg.generic {indexing_maps = [#id2, #id2],
                       iterator_types = ["parallel", "parallel"]}
      ins(%co : tensor<17x768xf32>) outs(%ge : tensor<17x768xf32>) {
  ^bb0(%in: f32, %o: f32):
    %q = arith.divf %in, %r2 : f32
    %er = math.erf %q : f32
    %p = arith.addf %er, %one : f32
    %h = arith.mulf %p, %half : f32
    %y = arith.mulf %in, %h : f32
    linalg.yield %y : f32
  } -> tensor<17x768xf32>
  return %g : tensor<17x768xf32>
}

// -----

// A **residual add** is still left alone, generic or not: its second operand is
// another activation read through a full-rank map, which is not something the
// call carries next to its output. Putting a requantization in front of one
// takes the `resadd_i8` away from the accelerator.

!qi32 = !quant.uniform<i32:f32, 1.500000e-05>

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @a_generic_residual_add_is_left_alone
// CHECK-NOT:     quant.qcast
func.func @a_generic_residual_add_is_left_alone(%a: tensor<17x192xi8>, %w: tensor<192x192xi8>,
                                                %skip: tensor<17x192xf32>)
    -> tensor<17x192xf32> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<17x192xi32>
  %init = linalg.fill ins(%z : i32) outs(%e : tensor<17x192xi32>) -> tensor<17x192xi32>
  %m = linalg.matmul {gemmlir.output_scale = 1.970000e-02 : f64}
      ins(%a, %w : tensor<17x192xi8>, tensor<192x192xi8>)
      outs(%init : tensor<17x192xi32>) -> tensor<17x192xi32>
  %s = quant.scast %m : tensor<17x192xi32> to tensor<17x192x!qi32>
  %d = quant.dcast %s : tensor<17x192x!qi32> to tensor<17x192xf32>
  %oe = tensor.empty() : tensor<17x192xf32>
  %r = linalg.generic {indexing_maps = [#id2, #id2, #id2],
                       iterator_types = ["parallel", "parallel"]}
      ins(%d, %skip : tensor<17x192xf32>, tensor<17x192xf32>) outs(%oe : tensor<17x192xf32>) {
  ^bb0(%in: f32, %sk: f32, %o: f32):
    %t = arith.addf %in, %sk : f32
    linalg.yield %t : f32
  } -> tensor<17x192xf32>
  return %r : tensor<17x192xf32>
}

// -----

// A **fan-out**, which is how a gate matrix arrives.
//
// An LSTM's gate matrix is one contraction chunked four ways, each quarter
// ending in a sigmoid or a tanh, so no single user of the tail blocks anything
// and the gates stayed in f32 -- **19.9 ms of that model's 37.6**. Following
// every branch and cutting when they all want it reaches them.
//
// The cut goes on **each branch**, below the fan-out, not above it. Above it the
// gate sum becomes a `resadd_i8`, which on 192 elements buys 3% and costs the
// same error this does; below it each gate gets a byte of its own and
// `--table-for-i8-elementwise` has all four. **37.6 -> 17.6 ms**, at 0.0038 ->
// 0.0066 relative L2.
//
// Every branch has to want it, because the cut is above all of them: one branch
// that needed the f32 would be paying for the others.

!qi32 = !quant.uniform<i32:f32, 4.000000e-05>

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// Each slice, then its own quantization -- not one above the two of them.
// CHECK-LABEL: func.func @a_fan_out_is_cut_branch_by_branch
// CHECK:         tensor.extract_slice
// CHECK:         quant.qcast
// CHECK:         tensor.extract_slice
// CHECK:         quant.qcast
func.func @a_fan_out_is_cut_branch_by_branch(%a: tensor<1x48xi8>, %w: tensor<48x96xi8>)
    -> (tensor<1x48xf32>, tensor<1x48xf32>) {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x96xi32>
  %init = linalg.fill ins(%z : i32) outs(%e : tensor<1x96xi32>) -> tensor<1x96xi32>
  %m = linalg.matmul {gemmlir.output_scale = 1.000000e-02 : f64}
      ins(%a, %w : tensor<1x48xi8>, tensor<48x96xi8>)
      outs(%init : tensor<1x96xi32>) -> tensor<1x96xi32>
  %s = quant.scast %m : tensor<1x96xi32> to tensor<1x96x!qi32>
  %d = quant.dcast %s : tensor<1x96x!qi32> to tensor<1x96xf32>
  // the chunk: two gates off one contraction
  %g0 = tensor.extract_slice %d[0, 0] [1, 48] [1, 1] : tensor<1x96xf32> to tensor<1x48xf32>
  %g1 = tensor.extract_slice %d[0, 48] [1, 48] [1, 1] : tensor<1x96xf32> to tensor<1x48xf32>
  %o = tensor.empty() : tensor<1x48xf32>
  %t0 = linalg.generic {indexing_maps = [#id2, #id2],
                        iterator_types = ["parallel", "parallel"]}
      ins(%g0 : tensor<1x48xf32>) outs(%o : tensor<1x48xf32>) {
  ^bb0(%in: f32, %x: f32):
    %v = math.tanh %in : f32
    linalg.yield %v : f32
  } -> tensor<1x48xf32>
  %t1 = linalg.generic {indexing_maps = [#id2, #id2],
                        iterator_types = ["parallel", "parallel"]}
      ins(%g1 : tensor<1x48xf32>) outs(%o : tensor<1x48xf32>) {
  ^bb0(%in: f32, %x: f32):
    %v = math.tanh %in : f32
    linalg.yield %v : f32
  } -> tensor<1x48xf32>
  return %t0, %t1 : tensor<1x48xf32>, tensor<1x48xf32>
}

// -----

// A **hard**-swish: `x * clamp(x + 3, 0, 6) / 6`. Every step of it is absorbable
// on its own and no branch holds a transcendental, so nothing here *stops* the
// tail -- and MobileNetV3-Small kept ten convolutions and eight depthwise
// convolutions in scalar loops because of it, 52 accelerator calls of 70.
//
// What the call cannot do is read its accumulator on **two** paths at once: the
// `mvout` writes each output element from one accumulator, once. That is the
// test, and it does not care what is on the branches. (A SiLU is the same shape
// and was already caught, but only by the `math.exp` on one side of it.)
//
// Note also where the clamp's 6 comes from: `nn.Hardswish` reaches torch-mlir
// with its bound as a rank-0 tensor, not a constant, which is why every test
// phrased in terms of the body's constants walked past this.

!qi32 = !quant.uniform<i32:f32, 4.000000e-05>

#chan = affine_map<(d0, d1, d2, d3) -> (d3)>
#id4  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#zero = affine_map<(d0, d1, d2, d3) -> ()>

// CHECK-LABEL: func.func @a_hard_swish_gets_a_requantization
// CHECK:         %[[B:[a-z0-9_]+]] = arith.addf %{{.*}} : tensor<1x8x8x16xf32>
// CHECK:         %[[Q:[a-z0-9_]+]] = quant.qcast %[[B]]
// CHECK-SAME:      to tensor<1x8x8x16x!quant.uniform<i8:f32, 1.000000e-02>>
// CHECK:         %[[S:[a-z0-9_]+]] = quant.scast %[[Q]]
// CHECK-SAME:      to tensor<1x8x8x16xi8>
// CHECK:         %[[W:[a-z0-9_]+]] = arith.sitofp %[[S]]
// CHECK:         arith.mulf %[[W]], %{{.*}}
func.func @a_hard_swish_gets_a_requantization(%acc: tensor<1x8x8x16xi8>,
                                              %flt: tensor<1x1x16x16xi8>,
                                              %bias: tensor<16xf32>,
                                              %six: tensor<f32>)
    -> tensor<1x8x8x16xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %three = arith.constant 3.000000e+00 : f32
  %sixf = arith.constant 6.000000e+00 : f32
  %e = tensor.empty() : tensor<1x8x8x16xi32>
  %be = tensor.empty() : tensor<1x8x8x16xf32>
  %f = linalg.generic {indexing_maps = [#chan, #id4],
                       iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%bias : tensor<16xf32>) outs(%be : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    linalg.yield %in : f32
  } -> tensor<1x8x8x16xf32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>,
                                 gemmlir.output_scale = 1.000000e-02 : f64}
      ins(%acc, %flt : tensor<1x8x8x16xi8>, tensor<1x1x16x16xi8>)
      outs(%e : tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
  %q = quant.scast %c : tensor<1x8x8x16xi32> to tensor<1x8x8x16x!qi32>
  %d = quant.dcast %q : tensor<1x8x8x16x!qi32> to tensor<1x8x8x16xf32>
  %b = arith.addf %d, %f : tensor<1x8x8x16xf32>
  // x + 3
  %s1 = tensor.empty() : tensor<1x8x8x16xf32>
  %a3 = linalg.generic {indexing_maps = [#id4, #id4],
                        iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%b : tensor<1x8x8x16xf32>) outs(%s1 : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %r = arith.addf %in, %three : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x16xf32>
  // max(_, 0)
  %s2 = tensor.empty() : tensor<1x8x8x16xf32>
  %rl = linalg.generic {indexing_maps = [#id4, #id4],
                        iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%a3 : tensor<1x8x8x16xf32>) outs(%s2 : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %p = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %p, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x16xf32>
  // min(_, 6), with the bound arriving as a rank-0 tensor
  %s3 = tensor.empty() : tensor<1x8x8x16xf32>
  %mn = linalg.generic {indexing_maps = [#id4, #zero, #id4],
                        iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%rl, %six : tensor<1x8x8x16xf32>, tensor<f32>)
      outs(%s3 : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %bnd: f32, %o: f32):
    %p = arith.cmpf olt, %in, %bnd : f32
    %r = arith.select %p, %in, %bnd : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x16xf32>
  // / 6
  %s4 = tensor.empty() : tensor<1x8x8x16xf32>
  %dv = linalg.generic {indexing_maps = [#id4, #id4],
                        iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%mn : tensor<1x8x8x16xf32>) outs(%s4 : tensor<1x8x8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %r = arith.divf %in, %sixf : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x16xf32>
  // and the join: the accumulator read a second time
  %s5 = tensor.empty() : tensor<1x8x8x16xf32>
  %out = linalg.generic {indexing_maps = [#id4, #id4, #id4],
                         iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%dv, %b : tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>)
      outs(%s5 : tensor<1x8x8x16xf32>) {
  ^bb0(%g: f32, %x: f32, %o: f32):
    %r = arith.mulf %g, %x : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x16xf32>
  return %out : tensor<1x8x8x16xf32>
}
