// What a frontend actually hands down. tosa.matmul survives the trip into the
// accelerator; see docs/pipeline.md for the ones that do not and why.

// RUN: gemmlir-opt --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))" %s \
// RUN: | gemmlir-opt --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
// RUN:               --buffer-deallocation-pipeline \
// RUN: | gemmlir-opt --convert-linalg-to-gemmlir | FileCheck %s

// tosa.matmul becomes linalg.batch_matmul, which becomes a loop of 2-D calls
// over subviews. The zero fill of the result proves there is nothing to
// accumulate, so the runtime gets a null bias.
// CHECK-LABEL: func.func @tosa_matmul
// CHECK:         scf.for
// CHECK:           gemmlir.matmul_i8
// CHECK-SAME:      {accumulate = false}
// CHECK-NOT:     linalg.batch_matmul
func.func @tosa_matmul(%a: tensor<1x32x64xi8>, %b: tensor<1x64x48xi8>) -> tensor<1x32x48xi32> {
  %azp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %bzp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %m = tosa.matmul %a, %b, %azp, %bzp
     : (tensor<1x32x64xi8>, tensor<1x64x48xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x32x48xi32>
  return %m : tensor<1x32x48xi32>
}
