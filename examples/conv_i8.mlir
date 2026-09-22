// A two-layer quantized block: two convolutions with fused bias, relu, scaling
// and max-pooling, joined by a saturating residual add. Nine linalg operations
// become four calls into the Gemmini runtime.
//
// Every buffer the accelerator touches is a function parameter, including the
// two intermediates. That is deliberate: on the board, handing the accelerator a
// freshly allocated output buffer on each call makes every other call wrong (see
// docs/pipeline.md, "reuse the buffers you hand the accelerator"). The i32
// accumulators below are local, but the fusion folds them away before anything
// reaches the hardware.

#nhwc = affine_map<(n, h, w, f) -> (n, h, w, f)>
#chan = affine_map<(n, h, w, f) -> (f)>
#hw   = affine_map<(d0, d1) -> (d0, d1)>

func.func @conv_block(
    %in:   memref<1x16x16x16xi8>,
    %flt1: memref<3x3x16x16xi8>, %bias1: memref<16xi32>,
    %flt2: memref<3x3x16x16xi8>,
    %window: memref<2x2xi8>,
    %tmp1: memref<1x7x7x16xi8>, %tmp2: memref<1x7x7x16xi8>,
    %out:  memref<1x7x7x16xi8>) {
  %zero = arith.constant 0 : i32
  %scale = arith.constant 2.500000e-02 : f32
  %relu_lo = arith.constant 0 : i32
  %sat_lo = arith.constant -128 : i32
  %sat_hi = arith.constant 127 : i32

  // --- layer 1: conv + bias + relu + requantize + 2x2 max-pool ---------------
  %acc1 = memref.alloc() : memref<1x14x14x16xi32>
  %req1 = memref.alloc() : memref<1x14x14x16xi8>
  linalg.fill ins(%zero : i32) outs(%acc1 : memref<1x14x14x16xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt1 : memref<1x16x16x16xi8>, memref<3x3x16x16xi8>)
    outs(%acc1 : memref<1x14x14x16xi32>)
  linalg.generic {indexing_maps = [#chan, #nhwc],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%bias1 : memref<16xi32>) outs(%acc1 : memref<1x14x14x16xi32>) {
  ^bb0(%b: i32, %a: i32):
    %s = arith.addi %a, %b : i32
    linalg.yield %s : i32
  }
  linalg.generic {indexing_maps = [#nhwc, #nhwc],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%acc1 : memref<1x14x14x16xi32>) outs(%req1 : memref<1x14x14x16xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %scale : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %lo = arith.maxsi %i, %relu_lo : i32
    %hi = arith.minsi %lo, %sat_hi : i32
    %t = arith.trunci %hi : i32 to i8
    linalg.yield %t : i8
  }
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%req1, %window : memref<1x14x14x16xi8>, memref<2x2xi8>)
    outs(%tmp1 : memref<1x7x7x16xi8>)
  memref.dealloc %acc1 : memref<1x14x14x16xi32>
  memref.dealloc %req1 : memref<1x14x14x16xi8>

  // --- layer 2: same shape, no bias -----------------------------------------
  %acc2 = memref.alloc() : memref<1x14x14x16xi32>
  %req2 = memref.alloc() : memref<1x14x14x16xi8>
  linalg.fill ins(%zero : i32) outs(%acc2 : memref<1x14x14x16xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt2 : memref<1x16x16x16xi8>, memref<3x3x16x16xi8>)
    outs(%acc2 : memref<1x14x14x16xi32>)
  linalg.generic {indexing_maps = [#nhwc, #nhwc],
                  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%acc2 : memref<1x14x14x16xi32>) outs(%req2 : memref<1x14x14x16xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %scale : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %lo = arith.maxsi %i, %sat_lo : i32
    %hi = arith.minsi %lo, %sat_hi : i32
    %t = arith.trunci %hi : i32 to i8
    linalg.yield %t : i8
  }
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%req2, %window : memref<1x14x14x16xi8>, memref<2x2xi8>)
    outs(%tmp2 : memref<1x7x7x16xi8>)
  memref.dealloc %acc2 : memref<1x14x14x16xi32>
  memref.dealloc %req2 : memref<1x14x14x16xi8>

  // --- residual add, saturating, over the flattened results ------------------
  %f1 = memref.collapse_shape %tmp1 [[0, 1], [2, 3]] : memref<1x7x7x16xi8> into memref<7x112xi8>
  %f2 = memref.collapse_shape %tmp2 [[0, 1], [2, 3]] : memref<1x7x7x16xi8> into memref<7x112xi8>
  %fo = memref.collapse_shape %out  [[0, 1], [2, 3]] : memref<1x7x7x16xi8> into memref<7x112xi8>
  linalg.generic {indexing_maps = [#hw, #hw, #hw], iterator_types = ["parallel", "parallel"]}
    ins(%f1, %f2 : memref<7x112xi8>, memref<7x112xi8>) outs(%fo : memref<7x112xi8>) {
  ^bb0(%p: i8, %q: i8, %o: i8):
    %pe = arith.extsi %p : i8 to i32
    %qe = arith.extsi %q : i8 to i32
    %sum = arith.addi %pe, %qe : i32
    %lo = arith.maxsi %sum, %sat_lo : i32
    %hi = arith.minsi %lo, %sat_hi : i32
    %t = arith.trunci %hi : i32 to i8
    linalg.yield %t : i8
  }
  return
}
