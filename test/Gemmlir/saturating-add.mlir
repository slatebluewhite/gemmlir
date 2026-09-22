// A saturating i8 add spelled out in arith -- extend, add, clamp, truncate --
// is what tiled_resadd_auto computes with unit scales, so it is offloaded.
// Plain linalg.add on i8 wraps and is left alone: on hardware those two answers
// differ in 1004 of 4096 elements for the inputs this was checked with.

// RUN: gemmlir-opt --convert-linalg-to-gemmlir %s | FileCheck %s

#id = affine_map<(d0,d1)->(d0,d1)>

// CHECK-LABEL: func.func @sat
// CHECK:         gemmlir.resadd_i8(%arg0, %arg1, %arg2)
// CHECK-NOT:     arith.addi
func.func @sat(%a: memref<64x64xi8>, %b: memref<64x64xi8>, %c: memref<64x64xi8>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id,#id,#id], iterator_types = ["parallel","parallel"]}
    ins(%a, %b : memref<64x64xi8>, memref<64x64xi8>) outs(%c : memref<64x64xi8>) {
  ^bb0(%x: i8, %y: i8, %o: i8):
    %xe = arith.extsi %x : i8 to i32
    %ye = arith.extsi %y : i8 to i32
    %s = arith.addi %xe, %ye : i32
    %m = arith.maxsi %s, %lo : i32
    %n = arith.minsi %m, %hi : i32
    %r = arith.trunci %n : i32 to i8
    linalg.yield %r : i8
  }
  return
}

// Clamping to [0, 127] instead is the same add with a relu on it.
// CHECK-LABEL: func.func @sat_relu
// CHECK:         gemmlir.resadd_i8(%arg0, %arg1, %arg2) {{.*}} {act = #gemmlir.act<relu>}
func.func @sat_relu(%a: memref<64x64xi8>, %b: memref<64x64xi8>, %c: memref<64x64xi8>) {
  %lo = arith.constant 0 : i32
  %hi = arith.constant 127 : i32
  linalg.generic {indexing_maps = [#id,#id,#id], iterator_types = ["parallel","parallel"]}
    ins(%a, %b : memref<64x64xi8>, memref<64x64xi8>) outs(%c : memref<64x64xi8>) {
  ^bb0(%x: i8, %y: i8, %o: i8):
    %xe = arith.extsi %x : i8 to i32
    %ye = arith.extsi %y : i8 to i32
    %s = arith.addi %xe, %ye : i32
    %m = arith.maxsi %s, %lo : i32
    %n = arith.minsi %m, %hi : i32
    %r = arith.trunci %n : i32 to i8
    linalg.yield %r : i8
  }
  return
}

// The clamp is the whole point: without it the add wraps, which the accelerator
// cannot do, so this one stays.
// CHECK-LABEL: func.func @wrapping
// CHECK-NOT:     gemmlir.
// CHECK:         arith.addi
func.func @wrapping(%a: memref<64x64xi8>, %b: memref<64x64xi8>, %c: memref<64x64xi8>) {
  linalg.generic {indexing_maps = [#id,#id,#id], iterator_types = ["parallel","parallel"]}
    ins(%a, %b : memref<64x64xi8>, memref<64x64xi8>) outs(%c : memref<64x64xi8>) {
  ^bb0(%x: i8, %y: i8, %o: i8):
    %s = arith.addi %x, %y : i8
    linalg.yield %s : i8
  }
  return
}

// A clamp to some other range is not this accelerator's saturation.
// CHECK-LABEL: func.func @other_range
// CHECK-NOT:     gemmlir.
// CHECK:         arith.addi
func.func @other_range(%a: memref<64x64xi8>, %b: memref<64x64xi8>, %c: memref<64x64xi8>) {
  %lo = arith.constant -100 : i32
  %hi = arith.constant 100 : i32
  linalg.generic {indexing_maps = [#id,#id,#id], iterator_types = ["parallel","parallel"]}
    ins(%a, %b : memref<64x64xi8>, memref<64x64xi8>) outs(%c : memref<64x64xi8>) {
  ^bb0(%x: i8, %y: i8, %o: i8):
    %xe = arith.extsi %x : i8 to i32
    %ye = arith.extsi %y : i8 to i32
    %s = arith.addi %xe, %ye : i32
    %m = arith.maxsi %s, %lo : i32
    %n = arith.minsi %m, %hi : i32
    %r = arith.trunci %n : i32 to i8
    linalg.yield %r : i8
  }
  return
}

// The scaled form, which is what a quantized residual block writes once
// --split-residual-add has taken the convolution's requantization out of it:
// two i8 tensors with a scale each, summed and requantized. The runtime applies
// A_scale and B_scale on the way in and C_scale on the way out, so the division
// by the output scale is folded into the two input scales and C_scale stays 1 --
// which matters, because MVIN_SCALE rounds *and clips to i8*, so a scale above
// one would saturate an operand before the sum.
// CHECK-LABEL: func.func @scaled
// CHECK:         gemmlir.resadd_i8(%arg0, %arg1, %arg2)
// CHECK-SAME:      lhs_scale = 2.500000e-01
// CHECK-SAME:      rhs_scale = 5.000000e-01
// CHECK-NOT:     out_scale
// CHECK-NOT:     arith.mulf
func.func @scaled(%a: memref<64x64xi8>, %b: memref<64x64xi8>, %c: memref<64x64xi8>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %sa = arith.constant 1.000000e-02 : f32
  %sb = arith.constant 2.000000e-02 : f32
  %so = arith.constant 4.000000e-02 : f32
  linalg.generic {indexing_maps = [#id,#id,#id], iterator_types = ["parallel","parallel"]}
    ins(%a, %b : memref<64x64xi8>, memref<64x64xi8>) outs(%c : memref<64x64xi8>) {
  ^bb0(%x: i8, %y: i8, %o: i8):
    %xf = arith.sitofp %x : i8 to f32
    %yf = arith.sitofp %y : i8 to f32
    %xs = arith.mulf %xf, %sa : f32
    %ys = arith.mulf %yf, %sb : f32
    %sum = arith.addf %xs, %ys : f32
    %q = arith.divf %sum, %so : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// A scale a frontend wrote as a multiply rather than a divide is the same
// thing, and a clamp to [0, 127] is the relu the narrowing already carries.
// CHECK-LABEL: func.func @scaled_by_multiply
// CHECK:         gemmlir.resadd_i8
// CHECK-SAME:      act = #gemmlir.act<relu>
// CHECK-SAME:      lhs_scale = 4.000000e+00
// CHECK-SAME:      rhs_scale = 4.000000e+00
func.func @scaled_by_multiply(%a: memref<64x64xi8>, %b: memref<64x64xi8>, %c: memref<64x64xi8>) {
  %lo = arith.constant 0 : i32
  %hi = arith.constant 127 : i32
  %s = arith.constant 2.000000e+00 : f32
  linalg.generic {indexing_maps = [#id,#id,#id], iterator_types = ["parallel","parallel"]}
    ins(%a, %b : memref<64x64xi8>, memref<64x64xi8>) outs(%c : memref<64x64xi8>) {
  ^bb0(%x: i8, %y: i8, %o: i8):
    %xf = arith.sitofp %x : i8 to f32
    %yf = arith.sitofp %y : i8 to f32
    %xs = arith.mulf %xf, %s : f32
    %ys = arith.mulf %yf, %s : f32
    %sum = arith.addf %xs, %ys : f32
    %q = arith.mulf %sum, %s : f32
    %r = math.roundeven %q : f32
    %i = arith.fptosi %r : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  }
  return
}

// Reading the same operand twice is not an add of two tensors.
// CHECK-LABEL: func.func @one_operand_twice
// CHECK-NOT:     gemmlir.resadd_i8
// CHECK:         linalg.generic
func.func @one_operand_twice(%a: memref<64x64xi8>, %b: memref<64x64xi8>, %c: memref<64x64xi8>) {
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %s = arith.constant 2.000000e-02 : f32
  linalg.generic {indexing_maps = [#id,#id,#id], iterator_types = ["parallel","parallel"]}
    ins(%a, %b : memref<64x64xi8>, memref<64x64xi8>) outs(%c : memref<64x64xi8>) {
  ^bb0(%x: i8, %y: i8, %o: i8):
    %xf = arith.sitofp %x : i8 to f32
    %xs = arith.mulf %xf, %s : f32
    %sum = arith.addf %xs, %xs : f32
    %r = math.roundeven %sum : f32
    %i = arith.fptosi %r : f32 to i32
    %cl = arith.maxsi %i, %lo : i32
    %ch = arith.minsi %cl, %hi : i32
    %t = arith.trunci %ch : i32 to i8
    linalg.yield %t : i8
  }
  return
}
