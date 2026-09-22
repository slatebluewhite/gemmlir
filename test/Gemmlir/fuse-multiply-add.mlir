// RUN: gemmlir-opt --fuse-multiply-add %s | FileCheck %s

// The dequantize tail: scale the accumulator, add the bias, activate.
// `fmadd.s` is one trip through the FPU where the pair is two, and on this
// in-order core the add waits for the multiply.

#acc = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func @dequantize_tail
// CHECK:         linalg.generic
// CHECK:           %[[F:.*]] = arith.sitofp
// CHECK:           math.fma %[[F]], %{{.*}}, %{{.*}} : f32
// CHECK-NOT:       arith.mulf
// CHECK-NOT:       arith.addf
func.func @dequantize_tail(%acc: tensor<8x16xi32>, %b: tensor<16xf32>,
                           %out: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %s = arith.constant 0.013 : f32
  %0 = linalg.generic {indexing_maps = [#acc, #bias, #acc],
                       iterator_types = ["parallel", "parallel"]}
      ins(%acc, %b: tensor<8x16xi32>, tensor<16xf32>) outs(%out: tensor<8x16xf32>) {
  ^bb0(%a: i32, %bb: f32, %o: f32):
    %f = arith.sitofp %a : i32 to f32
    %m = arith.mulf %f, %s : f32
    %p = arith.addf %m, %bb : f32
    linalg.yield %p : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// The multiply on either side of the add.

// CHECK-LABEL: func @addend_first
// CHECK:         math.fma %arg1, %arg2, %arg0 : f32
func.func @addend_first(%c: f32, %a: f32, %b: f32) -> f32 {
  %m = arith.mulf %a, %b : f32
  %r = arith.addf %c, %m : f32
  return %r : f32
}

// A multiply something else reads has to stay where it is, so folding a copy
// of it into the add would cost an instruction rather than save one.

// CHECK-LABEL: func @multiply_read_twice
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK-NOT:     math.fma
func.func @multiply_read_twice(%a: f32, %b: f32, %c: f32) -> (f32, f32) {
  %m = arith.mulf %a, %b : f32
  %r = arith.addf %m, %c : f32
  return %r, %m : f32, f32
}

// With a multiply on both sides only one of them can be fused; the other
// stays an operand of it.

// CHECK-LABEL: func @two_multiplies
// CHECK:         %[[M:.*]] = arith.mulf %arg2, %arg3
// CHECK:         math.fma %arg0, %arg1, %[[M]] : f32
func.func @two_multiplies(%a: f32, %b: f32, %c: f32, %d: f32) -> f32 {
  %m = arith.mulf %a, %b : f32
  %n = arith.mulf %c, %d : f32
  %r = arith.addf %m, %n : f32
  return %r : f32
}

// An add with no multiply under it.

// CHECK-LABEL: func @plain_add
// CHECK:         arith.addf
// CHECK-NOT:     math.fma
func.func @plain_add(%a: f32, %b: f32) -> f32 {
  %r = arith.addf %a, %b : f32
  return %r : f32
}

// It is not limited to f32 -- whatever the FPU has an fma for.

// CHECK-LABEL: func @double
// CHECK:         math.fma %arg0, %arg1, %arg2 : f64
func.func @double(%a: f64, %b: f64, %c: f64) -> f64 {
  %m = arith.mulf %a, %b : f64
  %r = arith.addf %m, %c : f64
  return %r : f64
}
