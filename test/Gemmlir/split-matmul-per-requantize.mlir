// RUN: gemmlir-opt --split-matmul-per-requantize --canonicalize --split-input-file %s | FileCheck %s

// A transformer's attention projection is one matmul writing Q, K and V side by
// side in one i32 accumulator, and three readers taking a third of the columns
// each at three different scales. One buffer, three scales, so the
// requantization cannot fold into the call and all three loops stay on the host.
//
// A column slice of the product is the same rows against a column slice of the
// weights, and `tiled_matmul_auto` reads its operands through a row stride --
// so the slice is a `memref.subview`, not a copy. Three calls, each with its
// own scale, each writing i8. What is left is the relayout the reader was doing
// anyway, now over bytes.

#acc2 = affine_map<(d0, d1) -> (d0, d1)>
#heads = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @qkv
// The accumulator and its dequantization are gone.
// CHECK-NOT:   gemmlir.matmul_i8(
// CHECK-NOT:   arith.sitofp
// Three calls on three slices of the weights. The scale is the reader's own
// folded with the dequantization above it: 2.0 * (1/2.0), 2.0 * (1/4.0),
// 2.0 * (1/8.0).
// CHECK:       %[[B0:.*]] = memref.subview %arg1[0, 0] [8, 4] [1, 1]
// CHECK:       gemmlir.matmul_i8_scale(%arg0, %[[B0]], {{.*}}) : {{.*}} -> memref<6x4xi8>
// CHECK-NOT:     scale =
// CHECK:       %[[B1:.*]] = memref.subview %arg1[0, 4] [8, 4] [1, 1]
// CHECK:       gemmlir.matmul_i8_scale(%arg0, %[[B1]], {{.*}}scale = 5.000000e-01 : f32}
// CHECK:       %[[B2:.*]] = memref.subview %arg1[0, 8] [8, 4] [1, 1]
// CHECK:       gemmlir.matmul_i8_scale(%arg0, %[[B2]], {{.*}}scale = 2.500000e-01 : f32}
// Each reader keeps its maps and its output and becomes a copy.
// CHECK:       linalg.generic
// CHECK-NEXT:  ^bb0(%[[IN:.*]]: i8, %{{.*}}: i8)
// CHECK-NEXT:  linalg.yield %[[IN]] : i8
func.func @qkv(%a: memref<6x8xi8>, %b: memref<8x12xi8>,
               %q: memref<1x2x6x2xi8>, %k: memref<1x2x6x2xi8>,
               %v: memref<1x2x6x2xi8>) {
  %deq = arith.constant 2.0 : f32
  %sq = arith.constant 2.0 : f32
  %sk = arith.constant 4.0 : f32
  %sv = arith.constant 8.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<6x12xi32>
  gemmlir.matmul_i8(%a, %b, %acc) : (memref<6x8xi8> x memref<8x12xi8>) -> memref<6x12xi32> {accumulate = false}
  %mid = memref.alloc() : memref<6x12xf32>
  linalg.generic {indexing_maps = [#acc2, #acc2], iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<6x12xi32>) outs(%mid : memref<6x12xf32>) {
  ^bb0(%in: i32, %out: f32):
    %f = arith.sitofp %in : i32 to f32
    %s = arith.mulf %f, %deq : f32
    linalg.yield %s : f32
  }
  %w = memref.expand_shape %mid [[0, 1], [2]] output_shape [1, 6, 12] : memref<6x12xf32> into memref<1x6x12xf32>
  %sq3 = memref.subview %w[0, 0, 0] [1, 6, 4] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x4xf32, strided<[72, 12, 1]>>
  %sk3 = memref.subview %w[0, 0, 4] [1, 6, 4] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x4xf32, strided<[72, 12, 1], offset: 4>>
  %sv3 = memref.subview %w[0, 0, 8] [1, 6, 4] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x4xf32, strided<[72, 12, 1], offset: 8>>
  %eq = memref.expand_shape %sq3 [[0], [1], [2, 3]] output_shape [1, 6, 2, 2] : memref<1x6x4xf32, strided<[72, 12, 1]>> into memref<1x6x2x2xf32, strided<[72, 12, 2, 1]>>
  %ek = memref.expand_shape %sk3 [[0], [1], [2, 3]] output_shape [1, 6, 2, 2] : memref<1x6x4xf32, strided<[72, 12, 1], offset: 4>> into memref<1x6x2x2xf32, strided<[72, 12, 2, 1], offset: 4>>
  %ev = memref.expand_shape %sv3 [[0], [1], [2, 3]] output_shape [1, 6, 2, 2] : memref<1x6x4xf32, strided<[72, 12, 1], offset: 8>> into memref<1x6x2x2xf32, strided<[72, 12, 2, 1], offset: 8>>
  linalg.generic {indexing_maps = [#heads, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%eq : memref<1x6x2x2xf32, strided<[72, 12, 2, 1]>>) outs(%q : memref<1x2x6x2xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %sq : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  linalg.generic {indexing_maps = [#heads, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ek : memref<1x6x2x2xf32, strided<[72, 12, 2, 1], offset: 4>>) outs(%k : memref<1x2x6x2xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %sk : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  linalg.generic {indexing_maps = [#heads, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ev : memref<1x6x2x2xf32, strided<[72, 12, 2, 1], offset: 8>>) outs(%v : memref<1x2x6x2xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %sv : f32
    %r = math.roundeven %d : f32
    %i = arith.fptosi %r : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %mid : memref<6x12xf32>
  memref.dealloc %acc : memref<6x12xi32>
  return
}

// -----

// A relu reader asks the accelerator for the activation, because RELU on the
// way out of `mvout` is a clamp to [0, 127] and that is exactly the lower bound
// the loop had.

#acc2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @relu_reader
// CHECK:       gemmlir.matmul_i8_scale({{.*}}act = #gemmlir.act<relu>, scale = 5.000000e-01 : f32}
// The second reader clamps at -128, so it asks for no activation -- which is
// the attribute's default and does not print.
// CHECK:       gemmlir.matmul_i8_scale({{.*}}) : {{.*}} {scale = 5.000000e-01 : f32}
func.func @relu_reader(%a: memref<6x8xi8>, %b: memref<8x12xi8>,
                       %x: memref<6x6xi8>, %y: memref<6x6xi8>) {
  %deq = arith.constant 2.0 : f32
  %s = arith.constant 4.0 : f32
  %zero = arith.constant 0 : i32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<6x12xi32>
  gemmlir.matmul_i8(%a, %b, %acc) : (memref<6x8xi8> x memref<8x12xi8>) -> memref<6x12xi32> {accumulate = false}
  %mid = memref.alloc() : memref<6x12xf32>
  linalg.generic {indexing_maps = [#acc2, #acc2], iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<6x12xi32>) outs(%mid : memref<6x12xf32>) {
  ^bb0(%in: i32, %out: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %deq : f32
    linalg.yield %m : f32
  }
  %w = memref.expand_shape %mid [[0, 1], [2]] output_shape [1, 6, 12] : memref<6x12xf32> into memref<1x6x12xf32>
  %l = memref.subview %w[0, 0, 0] [1, 6, 6] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x6xf32, strided<[72, 12, 1]>>
  %r = memref.subview %w[0, 0, 6] [1, 6, 6] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x6xf32, strided<[72, 12, 1], offset: 6>>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%l : memref<1x6x6xf32, strided<[72, 12, 1]>>) outs(%x : memref<6x6xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %zero : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%r : memref<1x6x6xf32, strided<[72, 12, 1], offset: 6>>) outs(%y : memref<6x6xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %mid : memref<6x12xf32>
  memref.dealloc %acc : memref<6x12xi32>
  return
}

// -----

// One reader is the case the existing fold already takes, and splitting it
// would only add a copy.

#acc2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @one_reader
// CHECK:       gemmlir.matmul_i8(
// CHECK-NOT:   gemmlir.matmul_i8_scale(
func.func @one_reader(%a: memref<6x8xi8>, %b: memref<8x12xi8>, %o: memref<6x12xi8>) {
  %deq = arith.constant 2.0 : f32
  %s = arith.constant 4.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<6x12xi32>
  gemmlir.matmul_i8(%a, %b, %acc) : (memref<6x8xi8> x memref<8x12xi8>) -> memref<6x12xi32> {accumulate = false}
  %mid = memref.alloc() : memref<6x12xf32>
  linalg.generic {indexing_maps = [#acc2, #acc2], iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<6x12xi32>) outs(%mid : memref<6x12xf32>) {
  ^bb0(%in: i32, %out: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %deq : f32
    linalg.yield %m : f32
  }
  linalg.generic {indexing_maps = [#acc2, #acc2], iterator_types = ["parallel", "parallel"]}
      ins(%mid : memref<6x12xf32>) outs(%o : memref<6x12xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %mid : memref<6x12xf32>
  memref.dealloc %acc : memref<6x12xi32>
  return
}

// -----

// A non-zero offset in the dequantization is not a scale, and the accelerator's
// output pipeline has nowhere to put it.

#acc2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @offset_refused
// CHECK:       gemmlir.matmul_i8(
// CHECK-NOT:   gemmlir.matmul_i8_scale(
func.func @offset_refused(%a: memref<6x8xi8>, %b: memref<8x12xi8>,
                          %x: memref<6x6xi8>, %y: memref<6x6xi8>) {
  %deq = arith.constant 2.0 : f32
  %off = arith.constant 1.5 : f32
  %s = arith.constant 4.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<6x12xi32>
  gemmlir.matmul_i8(%a, %b, %acc) : (memref<6x8xi8> x memref<8x12xi8>) -> memref<6x12xi32> {accumulate = false}
  %mid = memref.alloc() : memref<6x12xf32>
  linalg.generic {indexing_maps = [#acc2, #acc2], iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<6x12xi32>) outs(%mid : memref<6x12xf32>) {
  ^bb0(%in: i32, %out: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %deq : f32
    %p = arith.addf %m, %off : f32
    linalg.yield %p : f32
  }
  %w = memref.expand_shape %mid [[0, 1], [2]] output_shape [1, 6, 12] : memref<6x12xf32> into memref<1x6x12xf32>
  %l = memref.subview %w[0, 0, 0] [1, 6, 6] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x6xf32, strided<[72, 12, 1]>>
  %r = memref.subview %w[0, 0, 6] [1, 6, 6] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x6xf32, strided<[72, 12, 1], offset: 6>>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%l : memref<1x6x6xf32, strided<[72, 12, 1]>>) outs(%x : memref<6x6xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%r : memref<1x6x6xf32, strided<[72, 12, 1], offset: 6>>) outs(%y : memref<6x6xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %mid : memref<6x12xf32>
  memref.dealloc %acc : memref<6x12xi32>
  return
}

// -----

// Two readers on the same columns would do the arithmetic twice.

#acc2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @overlap_refused
// CHECK:       gemmlir.matmul_i8(
// CHECK-NOT:   gemmlir.matmul_i8_scale(
func.func @overlap_refused(%a: memref<6x8xi8>, %b: memref<8x12xi8>,
                           %x: memref<6x8xi8>, %y: memref<6x8xi8>) {
  %deq = arith.constant 2.0 : f32
  %s = arith.constant 4.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<6x12xi32>
  gemmlir.matmul_i8(%a, %b, %acc) : (memref<6x8xi8> x memref<8x12xi8>) -> memref<6x12xi32> {accumulate = false}
  %mid = memref.alloc() : memref<6x12xf32>
  linalg.generic {indexing_maps = [#acc2, #acc2], iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<6x12xi32>) outs(%mid : memref<6x12xf32>) {
  ^bb0(%in: i32, %out: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %deq : f32
    linalg.yield %m : f32
  }
  %w = memref.expand_shape %mid [[0, 1], [2]] output_shape [1, 6, 12] : memref<6x12xf32> into memref<1x6x12xf32>
  %l = memref.subview %w[0, 0, 0] [1, 6, 8] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x8xf32, strided<[72, 12, 1]>>
  %r = memref.subview %w[0, 0, 4] [1, 6, 8] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x8xf32, strided<[72, 12, 1], offset: 4>>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%l : memref<1x6x8xf32, strided<[72, 12, 1]>>) outs(%x : memref<6x8xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%r : memref<1x6x8xf32, strided<[72, 12, 1], offset: 4>>) outs(%y : memref<6x8xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %s : f32
    %rd = math.roundeven %d : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %mid : memref<6x12xf32>
  memref.dealloc %acc : memref<6x12xi32>
  return
}

// -----

// An explicit bias is a row the runtime broadcasts down the matrix, so it gets
// the same column slice the weights did.

#acc2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @bias_is_sliced
// CHECK-DAG:   memref.subview %arg2[0, 0] [1, 6] [1, 1] : memref<1x12xi32>
// CHECK-DAG:   memref.subview %arg2[0, 6] [1, 6] [1, 1] : memref<1x12xi32>
// CHECK:       gemmlir.matmul_i8_scale({{.*}}) bias(
func.func @bias_is_sliced(%a: memref<6x8xi8>, %b: memref<8x12xi8>, %d: memref<1x12xi32>,
                          %x: memref<6x6xi8>, %y: memref<6x6xi8>) {
  %deq = arith.constant 2.0 : f32
  %s = arith.constant 4.0 : f32
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %acc = memref.alloc() : memref<6x12xi32>
  gemmlir.matmul_i8(%a, %b, %acc) bias(%d : memref<1x12xi32>) : (memref<6x8xi8> x memref<8x12xi8>) -> memref<6x12xi32> {accumulate = false}
  %mid = memref.alloc() : memref<6x12xf32>
  linalg.generic {indexing_maps = [#acc2, #acc2], iterator_types = ["parallel", "parallel"]}
      ins(%acc : memref<6x12xi32>) outs(%mid : memref<6x12xf32>) {
  ^bb0(%in: i32, %out: f32):
    %f = arith.sitofp %in : i32 to f32
    %m = arith.mulf %f, %deq : f32
    linalg.yield %m : f32
  }
  %w = memref.expand_shape %mid [[0, 1], [2]] output_shape [1, 6, 12] : memref<6x12xf32> into memref<1x6x12xf32>
  %l = memref.subview %w[0, 0, 0] [1, 6, 6] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x6xf32, strided<[72, 12, 1]>>
  %r = memref.subview %w[0, 0, 6] [1, 6, 6] [1, 1, 1] : memref<1x6x12xf32> to memref<1x6x6xf32, strided<[72, 12, 1], offset: 6>>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%l : memref<1x6x6xf32, strided<[72, 12, 1]>>) outs(%x : memref<6x6xi8>) {
  ^bb0(%in: f32, %out: i8):
    %dv = arith.divf %in, %s : f32
    %rd = math.roundeven %dv : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%r : memref<1x6x6xf32, strided<[72, 12, 1], offset: 6>>) outs(%y : memref<6x6xi8>) {
  ^bb0(%in: f32, %out: i8):
    %dv = arith.divf %in, %s : f32
    %rd = math.roundeven %dv : f32
    %i = arith.fptosi %rd : f32 to i32
    %c0 = arith.maxsi %i, %lo : i32
    %c1 = arith.minsi %c0, %hi : i32
    %t = arith.trunci %c1 : i32 to i8
    linalg.yield %t : i8
  }
  memref.dealloc %mid : memref<6x12xf32>
  memref.dealloc %acc : memref<6x12xi32>
  return
}
