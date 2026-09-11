// RUN: gemmlir-opt --force-quantized-matmul %s | FileCheck %s

// f32 tensor matmul is rewritten into qcast -> i8 storage -> i8xi8->i32 matmul -> dcast.
// CHECK-LABEL: func.func @matmul_f32
// CHECK:         %[[QA:.*]] = quant.qcast %arg0 : tensor<128x128xf32> to tensor<128x128x!quant.uniform<i8:f32, 2.000000e-02>>
// CHECK:         %[[QB:.*]] = quant.qcast %arg1 : tensor<128x256xf32> to tensor<128x256x!quant.uniform<i8:f32, 2.000000e-02>>
// CHECK:         %[[A8:.*]] = quant.scast %[[QA]] : {{.*}} to tensor<128x128xi8>
// CHECK:         %[[B8:.*]] = quant.scast %[[QB]] : {{.*}} to tensor<128x256xi8>
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<128x256xi32>
// CHECK:         %[[Z:.*]] = linalg.fill ins(%{{.*}} : i32) outs(%[[E]] : tensor<128x256xi32>)
// CHECK:         %[[M:.*]] = linalg.matmul ins(%[[A8]], %[[B8]] : tensor<128x128xi8>, tensor<128x256xi8>) outs(%[[Z]] : tensor<128x256xi32>)
// CHECK:         %[[QM:.*]] = quant.scast %[[M]] : tensor<128x256xi32> to tensor<128x256x!quant.uniform<i32:f32, 4.000000e-04>>
// CHECK:         %[[R:.*]] = quant.dcast %[[QM]] : {{.*}} to tensor<128x256xf32>
// CHECK:         return %[[R]]
func.func @matmul_f32(%A: tensor<128x128xf32>, %B: tensor<128x256xf32>, %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x256xf32>) outs(%C : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}
