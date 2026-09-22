// The quant dialect lowers a qcast to fptosi(divf(x, scale)), which gets a
// quantization wrong twice.
//
// arith.fptosi truncates toward zero where quantization rounds to nearest; on
// the board that difference was the whole gap between a relative L2 error of
// 0.036 and 0.012. And it does not saturate: an activation past the calibrated
// range is undefined behaviour that *wraps* on RISC-V, turning +128 into -128.
// Driving the CNN's input up to 6x its calibrated range, the wrapping build's
// error against the same model in f32 jumped around erratically (0.43, 0.88,
// 0.57, 0.65, 0.72) while the saturating one degraded monotonically and was
// better everywhere (0.25, 0.35, 0.45, 0.53, 0.59). In range the two are
// bit-identical.

// RUN: gemmlir-opt --round-quantized-casts %s | FileCheck %s

// The clamp is done in i32 and then narrowed, which is the arithmetic the
// accelerator does -- gemmini.h scales the i32 accumulator and clips it to
// elem_t -- and is the shape --convert-linalg-to-gemmlir reads as a
// requantization.
// CHECK-LABEL: func.func @quantize
// CHECK-DAG:     %[[LO:.*]] = arith.constant dense<-128> : tensor<4x4xi32>
// CHECK-DAG:     %[[HI:.*]] = arith.constant dense<127> : tensor<4x4xi32>
// CHECK:         %[[D:.*]] = arith.divf
// CHECK-NEXT:    %[[R:.*]] = math.roundeven %[[D]]
// CHECK-NEXT:    %[[W:.*]] = arith.fptosi %[[R]] : tensor<4x4xf32> to tensor<4x4xi32>
// CHECK-NEXT:    %[[A:.*]] = arith.maxsi %[[W]], %[[LO]]
// CHECK-NEXT:    %[[B:.*]] = arith.minsi %[[A]], %[[HI]]
// CHECK-NEXT:    arith.trunci %[[B]] : tensor<4x4xi32> to tensor<4x4xi8>
func.func @quantize(%x: tensor<4x4xf32>, %scale: tensor<4x4xf32>) -> tensor<4x4xi8> {
  %d = arith.divf %x, %scale : tensor<4x4xf32>
  %i = arith.fptosi %d : tensor<4x4xf32> to tensor<4x4xi8>
  return %i : tensor<4x4xi8>
}

// Already rounded: it still gets the clamp, and in particular no second
// roundeven. The rewrite's own output converts to i32 over a roundeven, which
// is what keeps this from matching itself forever.
// CHECK-LABEL: func.func @already_rounded
// CHECK:         math.roundeven
// CHECK-NOT:     math.roundeven
// CHECK:         arith.trunci
func.func @already_rounded(%x: tensor<4x4xf32>, %scale: tensor<4x4xf32>) -> tensor<4x4xi8> {
  %d = arith.divf %x, %scale : tensor<4x4xf32>
  %r = math.roundeven %d : tensor<4x4xf32>
  %i = arith.fptosi %r : tensor<4x4xf32> to tensor<4x4xi8>
  return %i : tensor<4x4xi8>
}

// A conversion the input asked for on its own keeps truncating, and keeps
// wrapping: it is not a quantization and the pass does not second-guess it.
// CHECK-LABEL: func.func @plain_truncation
// CHECK-NOT:     math.roundeven
// CHECK:         arith.fptosi %{{.*}} : tensor<4x4xf32> to tensor<4x4xi8>
func.func @plain_truncation(%x: tensor<4x4xf32>) -> tensor<4x4xi8> {
  %i = arith.fptosi %x : tensor<4x4xf32> to tensor<4x4xi8>
  return %i : tensor<4x4xi8>
}

// An i32 storage type is already wide enough to hold what fptosi produces, so
// only the rounding is restored.
// CHECK-LABEL: func.func @wide_storage
// CHECK:         math.roundeven
// CHECK-NEXT:    arith.fptosi %{{.*}} : tensor<4x4xf32> to tensor<4x4xi32>
// CHECK-NOT:     arith.trunci
func.func @wide_storage(%x: tensor<4x4xf32>, %scale: tensor<4x4xf32>) -> tensor<4x4xi32> {
  %d = arith.divf %x, %scale : tensor<4x4xf32>
  %i = arith.fptosi %d : tensor<4x4xf32> to tensor<4x4xi32>
  return %i : tensor<4x4xi32>
}
