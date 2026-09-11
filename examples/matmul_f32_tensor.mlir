module {
  func.func @matmul_example(
      %A: tensor<128x128xf32>,
      %B: tensor<128x256xf32>,
      %C: tensor<128x256xf32>
  ) -> tensor<128x256xf32> {
    %res = linalg.matmul
              ins(%A, %B :
                  tensor<128x128xf32>, tensor<128x256xf32>)
             outs(%C :
                  tensor<128x256xf32>)
           -> tensor<128x256xf32>
    return %res : tensor<128x256xf32>
  }
}
