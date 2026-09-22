// RUN: gemmlir-opt --separate-max-pool --split-input-file %s | FileCheck %s

// `max` is associative and commutative, so a 3x3 window is three rows of three
// columns: eight comparisons an output become four, and nine word loads become
// three and three, because the horizontal result is shared between vertically
// adjacent outputs. Measured at GoogLeNet's shapes: -65% to -68%, identical
// outputs -- for a maximum there is no rounding to reassociate.

// CHECK-LABEL: func.func @three_by_three
// The row buffer keeps every input row and narrows the columns, and starts at
// the same identity the output did:
// CHECK:       %[[ROWS:.*]] = memref.alloc() {{.*}} memref<1x30x28x64xi8>
// CHECK:       linalg.fill ins(%{{.*}} : i8) outs(%[[ROWS]]
// CHECK:       %[[WH:.*]] = memref.alloc() {{.*}} memref<1x3xi8>
// CHECK:       %[[WV:.*]] = memref.alloc() {{.*}} memref<3x1xi8>
// CHECK:       linalg.pooling_nhwc_max
// CHECK-SAME:    ins(%arg0, %[[WH]]
// CHECK-SAME:    outs(%[[ROWS]]
// CHECK:       linalg.pooling_nhwc_max
// CHECK-SAME:    ins(%[[ROWS]], %[[WV]]
// CHECK-SAME:    outs(%arg2
func.func @three_by_three(%in: memref<1x30x30x64xi8>, %win: memref<3x3xi8>,
                          %out: memref<1x28x28x64xi8>) {
  %lo = arith.constant -128 : i8
  linalg.fill ins(%lo : i8) outs(%out : memref<1x28x28x64xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%in, %win : memref<1x30x30x64xi8>, memref<3x3xi8>)
      outs(%out : memref<1x28x28x64xi8>)
  return
}

// -----

// **Stride two is a loss and is refused.** The row pass computes every row of
// the padded input; the direct form visits `outH * kh` of them, and at stride
// two that is already the smaller number. Measured with them in:
// `squeezenet1_1`, whose pools are all stride 2, read +14.1%.

// CHECK-LABEL: func.func @stride_two_refused
// CHECK-NOT:   memref.alloc
// CHECK:       linalg.pooling_nhwc_max
// CHECK-NOT:   linalg.pooling_nhwc_max
func.func @stride_two_refused(%in: memref<1x32x32x8xi32>, %win: memref<3x3xi32>,
                              %out: memref<1x15x15x8xi32>) {
  %lo = arith.constant -2147483648 : i32
  linalg.fill ins(%lo : i32) outs(%out : memref<1x15x15x8xi32>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%in, %win : memref<1x32x32x8xi32>, memref<3x3xi32>)
      outs(%out : memref<1x15x15x8xi32>)
  return
}

// -----

// A dilation splits the same way a window does: the columns keep the
// horizontal one and the rows keep the vertical one.

// CHECK-LABEL: func.func @dilated
// CHECK:       linalg.pooling_nhwc_max
// CHECK-SAME:    dilations = dense<[1, 2]> : tensor<2xi64>
// CHECK-SAME:    strides = dense<1> : tensor<2xi64>
// CHECK:       linalg.pooling_nhwc_max
// CHECK-SAME:    dilations = dense<[2, 1]> : tensor<2xi64>
// CHECK-SAME:    strides = dense<1> : tensor<2xi64>
func.func @dilated(%in: memref<1x32x32x8xi32>, %win: memref<3x3xi32>,
                   %out: memref<1x28x28x8xi32>) {
  %lo = arith.constant -2147483648 : i32
  linalg.fill ins(%lo : i32) outs(%out : memref<1x28x28x8xi32>)
  linalg.pooling_nhwc_max {dilations = dense<2> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%in, %win : memref<1x32x32x8xi32>, memref<3x3xi32>)
      outs(%out : memref<1x28x28x8xi32>)
  return
}

// -----

// A 2x2 window saves one comparison of three and costs a whole buffer, so it
// stays as it is.

// CHECK-LABEL: func.func @two_by_two_refused
// CHECK-NOT:   memref.alloc
// CHECK:       linalg.pooling_nhwc_max
// CHECK-SAME:    memref<2x2xi8>
func.func @two_by_two_refused(%in: memref<1x16x16x32xi8>, %win: memref<2x2xi8>,
                              %out: memref<1x8x8x32xi8>) {
  %lo = arith.constant -128 : i8
  linalg.fill ins(%lo : i8) outs(%out : memref<1x8x8x32xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%in, %win : memref<1x16x16x32xi8>, memref<2x2xi8>)
      outs(%out : memref<1x8x8x32xi8>)
  return
}

// -----

// A window with only one row is already one-dimensional.

// CHECK-LABEL: func.func @one_row_refused
// CHECK-NOT:   memref.alloc
// CHECK:       linalg.pooling_nhwc_max
// CHECK-SAME:    memref<1x5xi8>
func.func @one_row_refused(%in: memref<1x8x20x32xi8>, %win: memref<1x5xi8>,
                           %out: memref<1x8x16x32xi8>) {
  %lo = arith.constant -128 : i8
  linalg.fill ins(%lo : i8) outs(%out : memref<1x8x16x32xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%in, %win : memref<1x8x20x32xi8>, memref<1x5xi8>)
      outs(%out : memref<1x8x16x32xi8>)
  return
}
