#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#chan = affine_map<(d0, d1, d2, d3) -> (d3)>
module {
  memref.global "private" constant @dwf : memref<2x2x2xi8> =
    dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> {alignment = 64 : i64}
  func.func @quantized_depthwise(%in: memref<1x5x5x2xi8>, %b: memref<2xi32>,
                                 %out: memref<1x4x4x2xi8>) {
    %z = arith.constant 0 : i32
    %zf = arith.constant 0.0 : f32
    %s = arith.constant 4.000000e-02 : f32
    %lo = arith.constant -128 : i32
    %hi = arith.constant 127 : i32
    %f = memref.get_global @dwf : memref<2x2x2xi8>
    %acc = memref.alloc() : memref<1x4x4x2xi32>
    linalg.fill ins(%z : i32) outs(%acc : memref<1x4x4x2xi32>)
    linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%in, %f : memref<1x5x5x2xi8>, memref<2x2x2xi8>) outs(%acc : memref<1x4x4x2xi32>)
    linalg.generic {indexing_maps = [#id4, #chan, #id4], iterator_types = ["parallel","parallel","parallel","parallel"]}
      ins(%acc, %b : memref<1x4x4x2xi32>, memref<2xi32>) outs(%out : memref<1x4x4x2xi8>) {
    ^bb0(%a: i32, %bb: i32, %o: i8):
      %sum = arith.addi %a, %bb : i32
      %fl = arith.sitofp %sum : i32 to f32
      %m = arith.mulf %fl, %s : f32
      %r = arith.maximumf %m, %zf : f32
      %n = math.roundeven %r : f32
      %i = arith.fptosi %n : f32 to i32
      %cl = arith.maxsi %i, %lo : i32
      %ch = arith.minsi %cl, %hi : i32
      %t = arith.trunci %ch : i32 to i8
      linalg.yield %t : i8
    }
    memref.dealloc %acc : memref<1x4x4x2xi32>
    return
  }
}
