// `tiled_conv_auto` always writes elem_t, so a linalg conv accumulating into i32
// has nothing to lower to on its own. The whole quantized chain is matched
// instead: conv into a local i32 temporary, an optional per-channel bias, and a
// requantization down to i8.
//
// linalg's conv has no padding (that is a separate pad on the input), so the
// offloaded call uses padding = 0.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s

#nhwc = affine_map<(n,h,w,f)->(n,h,w,f)>
#chan = affine_map<(n,h,w,f)->(f)>

// CHECK-LABEL: func.func @qconv
// CHECK:         gemmlir.conv2d_i8(%arg0, %arg1, %arg3) bias(%arg2 : memref<32xi32>)
// CHECK-SAME:    {act = #gemmlir.act<relu>, scale = 2.500000e-02 : f32}
// CHECK-NOT:     linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:     linalg.generic
func.func @qconv(%in: memref<1x16x16x16xi8>, %flt: memref<3x3x16x32xi8>,
                 %b: memref<32xi32>, %out: memref<1x14x14x32xi8>) {
  %z = arith.constant 0 : i32
  %s = arith.constant 2.500000e-02 : f32
  %lo = arith.constant 0 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<1x14x14x32xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<1x14x14x32xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt : memref<1x16x16x16xi8>, memref<3x3x16x32xi8>)
    outs(%acc : memref<1x14x14x32xi32>)
  linalg.generic {indexing_maps = [#chan, #nhwc], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%b : memref<32xi32>) outs(%acc : memref<1x14x14x32xi32>) {
  ^bb0(%bb: i32, %a: i32):
    %t = arith.addi %a, %bb : i32
    linalg.yield %t : i32
  }
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc : memref<1x14x14x32xi32>) outs(%out : memref<1x14x14x32xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %x = arith.maxsi %i, %lo : i32
    %y = arith.minsi %x, %hi : i32
    %t = arith.trunci %y : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<1x14x14x32xi32>
  return
}

// Without the requantization there is nothing to offload into, so the conv
// stays as it is -- an i32 convolution is not something the runtime can write.
// CHECK-LABEL: func.func @plain_conv
// CHECK-NOT:     gemmlir.
// CHECK:         linalg.conv_2d_nhwc_hwcf
func.func @plain_conv(%in: memref<1x16x16x16xi8>, %flt: memref<3x3x16x32xi8>,
                      %acc: memref<1x14x14x32xi32>) {
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt : memref<1x16x16x16xi8>, memref<3x3x16x32xi8>)
    outs(%acc : memref<1x14x14x32xi32>)
  return
}

// The runtime takes one stride for both axes.
// CHECK-LABEL: func.func @uneven_strides
// CHECK-NOT:     gemmlir.
// CHECK:         linalg.conv_2d_nhwc_hwcf
func.func @uneven_strides(%in: memref<1x16x16x16xi8>, %flt: memref<3x3x16x32xi8>,
                          %out: memref<1x7x14x32xi8>) {
  %z = arith.constant 0 : i32
  %s = arith.constant 2.500000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<1x7x14x32xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<1x7x14x32xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>}
    ins(%in, %flt : memref<1x16x16x16xi8>, memref<3x3x16x32xi8>)
    outs(%acc : memref<1x7x14x32xi32>)
  linalg.generic {indexing_maps = [#nhwc, #nhwc], iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc : memref<1x7x14x32xi32>) outs(%out : memref<1x7x14x32xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %x = arith.maxsi %i, %lo : i32
    %y = arith.minsi %x, %hi : i32
    %t = arith.trunci %y : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<1x7x14x32xi32>
  return
}
