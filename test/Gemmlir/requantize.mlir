// A requantization sitting on an i32 matmul folds into matmul_i8_scale, which
// does the same thing in the mvout pipeline.
//
// The `math.roundeven` is load-bearing: the accelerator's mvout scaling rounds
// half to even (measured on hardware -- it disagrees with truncation on every
// exact .5), so a requantize that truncates means something else and is left in
// software. For the inputs this was checked with, the two roundings gave
// different answers in 790 of 1536 elements.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s

#id = affine_map<(d0,d1)->(d0,d1)>

// CHECK-LABEL: func.func @quant
// CHECK:         gemmlir.matmul_i8_scale(%arg0, %arg1, %arg2)
// CHECK-SAME:    {scale = 5.000000e-02 : f32}
// CHECK-NOT:     linalg.generic
func.func @quant(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi8>) {
  %z = arith.constant 0 : i32
  %s = arith.constant 5.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<32x48xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>) outs(%acc : memref<32x48xi32>)
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<32x48xi32>) outs(%C : memref<32x48xi8>) {
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
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// Clamping to [0, 127] instead folds the relu in too.
// CHECK-LABEL: func.func @quant_relu
// CHECK:         gemmlir.matmul_i8_scale({{.*}}) {{.*}} {act = #gemmlir.act<relu>, scale = 5.000000e-02 : f32}
func.func @quant_relu(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi8>) {
  %z = arith.constant 0 : i32
  %s = arith.constant 5.000000e-02 : f32
  %lo = arith.constant 0 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<32x48xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>) outs(%acc : memref<32x48xi32>)
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<32x48xi32>) outs(%C : memref<32x48xi8>) {
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
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// No roundeven: truncation is not what the accelerator does, so this stays.
// CHECK-LABEL: func.func @trunc_round
// CHECK:         gemmlir.matmul_i8(
// CHECK:         linalg.generic
func.func @trunc_round(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi8>) {
  %z = arith.constant 0 : i32
  %s = arith.constant 5.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<32x48xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>) outs(%acc : memref<32x48xi32>)
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<32x48xi32>) outs(%C : memref<32x48xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %i = arith.fptosi %m : f32 to i32
    %x = arith.maxsi %i, %lo : i32
    %y = arith.minsi %x, %hi : i32
    %t = arith.trunci %y : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// What the --quantize pipeline actually produces for a convolution layer, once
// the bias has moved onto the accumulator: an i32 bias, the scaling split into
// the dequantize `mulf s1` and the requantize `divf s2`, and a relu in float
// between them. Each of those is something the mvout pipeline does anyway --
// the hardware multiplies by one scale, so s1/s2 is the only form it has, and
// round(max(x,0)) == max(round(x),0) makes hoisting the relu to `lo = 0` exact.
// 2.0e-3 / 5.0e-2 = 0.04, to whatever f32 makes of dividing those two.
// CHECK-LABEL: func.func @quant_bias_relu
// CHECK:         %[[D:.*]] = memref.expand_shape %arg3
// CHECK-SAME:      memref<48xi32> into memref<1x48xi32>
// CHECK:         gemmlir.matmul_i8_scale(%arg0, %arg1, %arg2) bias(%[[D]] : memref<1x48xi32>)
// CHECK-SAME:    {act = #gemmlir.act<relu>, scale = 0.040000{{[0-9]*}} : f32}
// CHECK-NOT:     linalg.generic
#col = affine_map<(d0,d1)->(d1)>
func.func @quant_bias_relu(%A: memref<32x64xi8>, %B: memref<64x48xi8>,
                           %C: memref<32x48xi8>, %bias: memref<48xi32>) {
  %z = arith.constant 0 : i32
  %zf = arith.constant 0.0 : f32
  %s1 = arith.constant 2.000000e-03 : f32
  %s2 = arith.constant 5.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<32x48xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>) outs(%acc : memref<32x48xi32>)
  linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc, %bias : memref<32x48xi32>, memref<48xi32>) outs(%C : memref<32x48xi8>) {
  ^bb0(%a: i32, %b: i32, %o: i8):
    %sum = arith.addi %a, %b : i32
    %f = arith.sitofp %sum : i32 to f32
    %m = arith.mulf %f, %s1 : f32
    %r = arith.maximumf %m, %zf : f32
    %d = arith.divf %r, %s2 : f32
    %n = math.roundeven %d : f32
    %i = arith.fptosi %n : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// A bias broadcast over the rows is not something the runtime's D can express
// (it repeats a row *down* the rows), so the layer stays in software.
// CHECK-LABEL: func.func @row_bias_stays
// CHECK:         gemmlir.matmul_i8(
// CHECK:         linalg.generic
#row = affine_map<(d0,d1)->(d0)>
func.func @row_bias_stays(%A: memref<32x64xi8>, %B: memref<64x48xi8>,
                          %C: memref<32x48xi8>, %bias: memref<32xi32>) {
  %z = arith.constant 0 : i32
  %s1 = arith.constant 2.000000e-03 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<32x48xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>) outs(%acc : memref<32x48xi32>)
  linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc, %bias : memref<32x48xi32>, memref<32xi32>) outs(%C : memref<32x48xi8>) {
  ^bb0(%a: i32, %b: i32, %o: i8):
    %sum = arith.addi %a, %b : i32
    %f = arith.sitofp %sum : i32 to f32
    %m = arith.mulf %f, %s1 : f32
    %n = math.roundeven %m : f32
    %i = arith.fptosi %n : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// A bounded activation -- ReLU6, which MobileNet is built from -- is an upper
// clamp in f32 before the quantization. The accelerator has no such activation,
// and does not need one when the bound is already outside what the quantization
// can represent: the saturation on the way out of the accumulator clips first.
// Here the output scale is 1/16 and the bound is 8, so the bound lands at 128 --
// past the 127 the mvout clips to -- and the clamp never fires.
// CHECK-LABEL: func.func @relu6_is_inert
// CHECK:         gemmlir.matmul_i8_scale
// CHECK-SAME:      act = #gemmlir.act<relu>
// CHECK-NOT:     arith.select
func.func @relu6_is_inert(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi8>) {
  %c0 = arith.constant 0 : i32
  %lo = arith.constant 0 : i32
  %hi = arith.constant 127 : i32
  %zero = arith.constant 0.0 : f32
  %six = arith.constant 8.000000e+00 : f32
  %s = arith.constant 6.250000e-02 : f32
  %acc = memref.alloc() : memref<32x48xi32>
  linalg.fill ins(%c0 : i32) outs(%acc : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>)
                outs(%acc : memref<32x48xi32>)
  linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
                  iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<32x48xi32>) outs(%C : memref<32x48xi8>) {
  ^bb0(%in: i32, %out: i8):
    %f = arith.sitofp %in : i32 to f32
    %g = arith.cmpf ogt, %f, %zero : f32
    %r = arith.select %g, %f, %zero : f32
    %l = arith.cmpf olt, %six, %r : f32
    %b = arith.select %l, %six, %r : f32
    %d = arith.divf %b, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// A bound that actually bites is a different function, and is left where it is.
// The same scale with a bound of 4 lands at 64, well inside the range.
// CHECK-LABEL: func.func @relu6_that_bites
// CHECK-NOT:     gemmlir.matmul_i8_scale
// CHECK:         gemmlir.matmul_i8(
// CHECK:         arith.select
func.func @relu6_that_bites(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi8>) {
  %c0 = arith.constant 0 : i32
  %lo = arith.constant 0 : i32
  %hi = arith.constant 127 : i32
  %zero = arith.constant 0.0 : f32
  %four = arith.constant 4.000000e+00 : f32
  %s = arith.constant 6.250000e-02 : f32
  %acc = memref.alloc() : memref<32x48xi32>
  linalg.fill ins(%c0 : i32) outs(%acc : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>)
                outs(%acc : memref<32x48xi32>)
  linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
                  iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<32x48xi32>) outs(%C : memref<32x48xi8>) {
  ^bb0(%in: i32, %out: i8):
    %f = arith.sitofp %in : i32 to f32
    %g = arith.cmpf ogt, %f, %zero : f32
    %r = arith.select %g, %f, %zero : f32
    %l = arith.cmpf olt, %four, %r : f32
    %b = arith.select %l, %four, %r : f32
    %d = arith.divf %b, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// A `hardtanh` clamps on both sides. The lower half goes the same way as the
// upper: the accumulator saturates at -128 anyway, so a bound that lands there
// or past it never fires. With the scale 1/16 below, -8 lands at -128 exactly.
// CHECK-LABEL: func.func @hardtanh_is_inert
// CHECK:         gemmlir.matmul_i8_scale
// CHECK-NOT:     arith.select
func.func @hardtanh_is_inert(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi8>) {
  %c0 = arith.constant 0 : i32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %low = arith.constant -8.000000e+00 : f32
  %high = arith.constant 8.000000e+00 : f32
  %s = arith.constant 6.250000e-02 : f32
  %acc = memref.alloc() : memref<32x48xi32>
  linalg.fill ins(%c0 : i32) outs(%acc : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>)
                outs(%acc : memref<32x48xi32>)
  linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
                  iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<32x48xi32>) outs(%C : memref<32x48xi8>) {
  ^bb0(%in: i32, %out: i8):
    %f = arith.sitofp %in : i32 to f32
    %a = arith.maximumf %f, %low : f32
    %b = arith.minimumf %a, %high : f32
    %d = arith.divf %b, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// A lower bound that bites is a real operation. -4 lands at -64, well inside.
// CHECK-LABEL: func.func @hardtanh_that_bites
// CHECK-NOT:     gemmlir.matmul_i8_scale
// CHECK:         gemmlir.matmul_i8(
// CHECK:         arith.maximumf
func.func @hardtanh_that_bites(%A: memref<32x64xi8>, %B: memref<64x48xi8>, %C: memref<32x48xi8>) {
  %c0 = arith.constant 0 : i32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %low = arith.constant -4.000000e+00 : f32
  %high = arith.constant 8.000000e+00 : f32
  %s = arith.constant 6.250000e-02 : f32
  %acc = memref.alloc() : memref<32x48xi32>
  linalg.fill ins(%c0 : i32) outs(%acc : memref<32x48xi32>)
  linalg.matmul ins(%A, %B : memref<32x64xi8>, memref<64x48xi8>)
                outs(%acc : memref<32x48xi32>)
  linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
                  iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<32x48xi32>) outs(%C : memref<32x48xi8>) {
  ^bb0(%in: i32, %out: i8):
    %f = arith.sitofp %in : i32 to f32
    %a = arith.maximumf %f, %low : f32
    %b = arith.minimumf %a, %high : f32
    %d = arith.divf %b, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// A second matmul between the first and its requantization is not a barrier as
// long as it only *reads* the operands the fold has to move down: a
// transformer's projections share one activation and are emitted back to back,
// so each of them reads what the others read.
// CHECK-LABEL: func.func @read_between
// CHECK:         gemmlir.matmul_i8(%arg0, %arg2, %{{.*}})
// CHECK:         gemmlir.matmul_i8_scale(%arg0, %arg1, %arg3)
// CHECK-NOT:     linalg.generic
func.func @read_between(%a: memref<32x64xi8>, %b: memref<64x48xi8>,
                        %b2: memref<64x48xi8>, %out: memref<32x48xi8>,
                        %other: memref<32x48xi32>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %s = arith.constant 2.000000e-02 : f32
  %acc = memref.alloc() : memref<32x48xi32>
  gemmlir.matmul_i8(%a, %b, %acc)
    : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi32> {accumulate = false}
  gemmlir.matmul_i8(%a, %b2, %other)
    : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi32> {accumulate = false}
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<32x48xi32>) outs(%out : memref<32x48xi8>) {
  ^bb0(%in: i32, %o: i8):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// Writing one of them in between is still a barrier.
// CHECK-LABEL: func.func @write_between
// CHECK:         gemmlir.matmul_i8(%arg0, %arg1, %{{.*}})
// CHECK:         linalg.generic
func.func @write_between(%a: memref<32x64xi8>, %b: memref<64x48xi8>,
                         %src: memref<64x48xi8>, %out: memref<32x48xi8>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %s = arith.constant 2.000000e-02 : f32
  %acc = memref.alloc() : memref<32x48xi32>
  gemmlir.matmul_i8(%a, %b, %acc)
    : (memref<32x64xi8> x memref<64x48xi8>) -> memref<32x48xi32> {accumulate = false}
  memref.copy %src, %b : memref<64x48xi8> to memref<64x48xi8>
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<32x48xi32>) outs(%out : memref<32x48xi8>) {
  ^bb0(%in: i32, %o: i8):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<32x48xi32>
  return
}

// An im2col contraction writes its accumulator as a matrix and the rest of the
// model reads it as the image it stands for, so the matmul holds a
// `memref.collapse_shape` of the temporary and the requantization reads the
// four-dimensional view. Those are the same bytes in the same order, and
// without walking through the reshape the whole tail of every packed
// convolution -- bias, relu and the requantize itself -- stayed a scalar loop:
// 9216 elements of `atr`, twice over.

#img4 = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>
#col4 = affine_map<(d0,d1,d2,d3)->(d3)>

// CHECK-LABEL: func.func @through_a_reshape
// CHECK:         %[[OUT:.*]] = memref.collapse_shape %arg2
// CHECK-SAME:      into memref<576x16xi8>
// CHECK:         gemmlir.matmul_i8_scale(%arg0, %arg1, %[[OUT]])
// CHECK-SAME:      bias(%{{.*}} : memref<1x16xi32>)
// CHECK-SAME:      {act = #gemmlir.act<relu>, scale = 5.000000e-02 : f32}
// CHECK-NOT:     linalg.generic
func.func @through_a_reshape(%A: memref<576x27xi8>, %B: memref<27x16xi8>,
                             %C: memref<1x24x24x16xi8>, %bias: memref<16xi32>) {
  %z = arith.constant 0 : i32
  %zf = arith.constant 0.0 : f32
  %s = arith.constant 5.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<1x24x24x16xi32>
  %flat = memref.collapse_shape %acc [[0, 1, 2], [3]]
    : memref<1x24x24x16xi32> into memref<576x16xi32>
  linalg.fill ins(%z : i32) outs(%flat : memref<576x16xi32>)
  linalg.matmul ins(%A, %B : memref<576x27xi8>, memref<27x16xi8>)
                outs(%flat : memref<576x16xi32>)
  linalg.generic {indexing_maps = [#img4, #col4, #img4],
                  iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc, %bias : memref<1x24x24x16xi32>, memref<16xi32>)
    outs(%C : memref<1x24x24x16xi8>) {
  ^bb0(%a: i32, %b: i32, %o: i8):
    %sum = arith.addi %a, %b : i32
    %f = arith.sitofp %sum : i32 to f32
    %m = arith.mulf %f, %s : f32
    %p = arith.cmpf ugt, %m, %zf : f32
    %r = arith.select %p, %m, %zf : f32
    %n = math.roundeven %r : f32
    %i = arith.fptosi %n : f32 to i32
    %c = arith.maxsi %i, %lo : i32
    %d = arith.minsi %c, %hi : i32
    %t = arith.trunci %d : i32 to i8
    linalg.yield %t : i8
  } 
  memref.dealloc %acc : memref<1x24x24x16xi32>
  return
}

// A *subview* is not the same bytes: the matmul would have written part of the
// buffer and the requantization reads all of it, so this one stays.

// CHECK-LABEL: func.func @through_a_subview
// CHECK:         linalg.generic
func.func @through_a_subview(%A: memref<576x27xi8>, %B: memref<27x16xi8>,
                             %C: memref<576x32xi8>) {
  %z = arith.constant 0 : i32
  %s = arith.constant 5.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<576x32xi32>
  %part = memref.subview %acc[0, 0] [576, 16] [1, 1]
    : memref<576x32xi32> to memref<576x16xi32, strided<[32, 1]>>
  linalg.fill ins(%z : i32) outs(%part : memref<576x16xi32, strided<[32, 1]>>)
  linalg.matmul ins(%A, %B : memref<576x27xi8>, memref<27x16xi8>)
                outs(%part : memref<576x16xi32, strided<[32, 1]>>)
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel"]}
    ins(%acc : memref<576x32xi32>) outs(%C : memref<576x32xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %n = math.roundeven %m : f32
    %i = arith.fptosi %n : f32 to i32
    %c = arith.maxsi %i, %lo : i32
    %d = arith.minsi %c, %hi : i32
    %t = arith.trunci %d : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<576x32xi32>
  return
}

// -----

// A bias of exactly zero. torchvision's SqueezeNet initialises every
// convolution's bias to zero, so the per-channel tensor is a splat and the
// frontend leaves `x + 0.0` in the requantization's tail. Stepping over it is
// what lets the convolution fold: without this, twenty-two of that model's
// twenty-seven contractions stayed scalar loops.
//
// `x + 0` differs from `x` only in the sign of a zero, and both convert to the
// integer 0.

#wholeA = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#id4A   = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @zero_bias_still_folds
// CHECK:         gemmlir.conv2d_i8
// CHECK-SAME:    act = #gemmlir.act<relu>
// CHECK-NOT:     linalg.conv_2d_nhwc_hwcf
func.func @zero_bias_still_folds(%in: memref<1x8x8x16xi8>, %flt: memref<3x3x16x16xi8>,
                                 %out: memref<1x6x6x16xi8>) {
  %z = arith.constant 0 : i32
  %zero = arith.constant 0.000000e+00 : f32
  %s = arith.constant 2.500000e-02 : f32
  %t = arith.constant 5.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<1x6x6x16xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<1x6x6x16xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt : memref<1x8x8x16xi8>, memref<3x3x16x16xi8>) outs(%acc : memref<1x6x6x16xi32>)
  linalg.generic {indexing_maps = [#wholeA, #id4A],
                  iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc : memref<1x6x6x16xi32>) outs(%out : memref<1x6x6x16xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %b = arith.addf %m, %zero : f32
    %c = arith.cmpf ugt, %b, %zero : f32
    %r = arith.select %c, %b, %zero : f32
    %d = arith.divf %r, %t : f32
    %e = math.roundeven %d : f32
    %i = arith.fptosi %e : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %q = arith.trunci %c1 : i32 to i8
    linalg.yield %q : i8
  }
  memref.dealloc %acc : memref<1x6x6x16xi32>
  return
}

// -----

// A bias that is not zero is a real one and the convolution does not fold: the
// accelerator's bias is one i32 per output channel, not a number added to
// everything in f32.

#wholeB = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#id4B   = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @real_bias_does_not
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:     gemmlir.conv2d_i8
func.func @real_bias_does_not(%in: memref<1x8x8x16xi8>, %flt: memref<3x3x16x16xi8>,
                              %out: memref<1x6x6x16xi8>) {
  %z = arith.constant 0 : i32
  %zero = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %s = arith.constant 2.500000e-02 : f32
  %t = arith.constant 5.000000e-02 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<1x6x6x16xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<1x6x6x16xi32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %flt : memref<1x8x8x16xi8>, memref<3x3x16x16xi8>) outs(%acc : memref<1x6x6x16xi32>)
  linalg.generic {indexing_maps = [#wholeB, #id4B],
                  iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%acc : memref<1x6x6x16xi32>) outs(%out : memref<1x6x6x16xi8>) {
  ^bb0(%a: i32, %o: i8):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %b = arith.addf %m, %one : f32
    %d = arith.divf %b, %t : f32
    %e = math.roundeven %d : f32
    %i = arith.fptosi %e : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %q = arith.trunci %c1 : i32 to i8
    linalg.yield %q : i8
  }
  memref.dealloc %acc : memref<1x6x6x16xi32>
  return
}

// -----

// The same fold, on a **batch** matmul. It becomes an `scf.for` over rank-reduced
// slices, so the accumulator is written a slice at a time while the
// requantization reads the whole of it from outside the loop -- neither a 2-D
// accumulator nor one the call holds directly, so the fold that handles an
// ordinary matmul saw nothing and a ConvNeXt block's two pointwise convolutions
// kept a full f32 pass over an i32 buffer. **82.55 -> 73.03 ms** on that block,
// same answer to the byte.
//
// Sinking it into the loop is sound because every slice gets the same treatment:
// one scale for the whole tensor, and a bias the runtime repeats down the rows,
// which is per-column and so already the same for every slice.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s --check-prefix=BATCH

#idb = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#colb = affine_map<(d0, d1, d2) -> (d2)>

// Each slice's call writes i8 straight into the output, and the requantization
// is gone. (The accumulator's allocation is left for --canonicalize, which has
// nothing but its own dealloc to look at by then.)
// BATCH-LABEL: func.func @batched_requantize
// BATCH:         scf.for
// BATCH:           %[[S:.*]] = memref.subview %arg2
// BATCH-SAME:        to memref<8x16xi8, strided<[16, 1], offset: ?>>
// BATCH:           gemmlir.matmul_i8_scale({{.*}}%[[S]])
// BATCH-NOT:     linalg.generic
func.func @batched_requantize(%a: memref<4x8x32xi8>, %b: memref<4x32x16xi8>,
                              %out: memref<4x8x16xi8>, %bias: memref<16xi32>) {
  %z = arith.constant 0 : i32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %s = arith.constant 2.500000e-02 : f32
  %acc = memref.alloc() {alignment = 64 : i64} : memref<4x8x16xi32>
  linalg.fill ins(%z : i32) outs(%acc : memref<4x8x16xi32>)
  linalg.batch_matmul ins(%a, %b : memref<4x8x32xi8>, memref<4x32x16xi8>)
                      outs(%acc : memref<4x8x16xi32>)
  linalg.generic {indexing_maps = [#idb, #colb, #idb],
                  iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%acc, %bias : memref<4x8x16xi32>, memref<16xi32>)
      outs(%out : memref<4x8x16xi8>) {
  ^bb0(%in: i32, %bb: i32, %o: i8):
    %d = arith.addi %in, %bb : i32
    %f = arith.sitofp %d : i32 to f32
    %m = arith.mulf %f, %s : f32
    %r = math.roundeven %m : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %acc : memref<4x8x16xi32>
  return
}
