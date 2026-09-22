// RUN: gemmlir-opt --pack-int8-max-pool --split-input-file %s | FileCheck %s

// A max-pool over i8 is **independent per channel**, and NHWC puts the channels
// next to each other -- so eight of them are one 64-bit word and the whole
// window can be walked eight at a time.
//
// The scalar form is what a program-counter profile finds at the top of
// GoogLeNet: **39% of the model** in nine byte loads and eight compares per
// output, each compare a data-dependent branch. Measured on the board on one of
// its shapes (14x14x256 in, 3x3, 12x12x256 out), 200 repetitions:
// **111.10 ms scalar against 21.65 ms packed, 5.13x** (`scripts/poolswar.c`).
//
// The byte-wise unsigned maximum, with every byte flipped into unsigned order
// once on the way in and once on the way out:
//
//     d   = (a | HI) - (b & ~HI)                 // bit 7: al >= bl
//     ge  = HI & ((a & ~b) | (~(a ^ b) & d))     // bit 7: a_i >= b_i
//     m   = ((ge >> 7) & LOW) * 0xFF
//     max = b ^ ((a ^ b) & m)
//
// Checked exhaustively -- all 65,536 byte pairs and 200,000 random word pairs.
// The obvious shorter formula is wrong on 248,893 of them.

// CHECK-LABEL: func.func @a_pool_is_packed
// The buffers are read as words, not bytes:
// CHECK-DAG:     %[[HI:.*]] = arith.constant -9187201950435737472 : i64
// CHECK-DAG:     %[[LOW:.*]] = arith.constant 72340172838076673 : i64
// CHECK:         %[[FS:.*]] = memref.collapse_shape %arg0
// CHECK:         %[[VS:.*]] = memref.view %[[FS]]{{.*}} to memref<6272xi64>
// CHECK:         %[[FO:.*]] = memref.collapse_shape %arg2
// CHECK:         %[[VO:.*]] = memref.view %[[FO]]{{.*}} to memref<4608xi64>
// The accumulator starts from what the output already holds, which is the
// operation's own semantics and leaves the fill in front of it alone:
// CHECK:         memref.load %[[VO]]
// CHECK:         arith.xori %{{.*}}, %[[HI]]
// CHECK:         scf.for {{.*}}iter_args
// CHECK:           scf.for {{.*}}iter_args
// CHECK:             memref.load %[[VS]]
// CHECK:             arith.subi
// CHECK:             arith.muli %{{.*}}, %{{.*}} : i64
// CHECK:         memref.store %{{.*}}, %[[VO]]
// CHECK-NOT:     linalg.pooling_nhwc_max

func.func @a_pool_is_packed(%in: memref<1x14x14x256xi8>, %w: memref<3x3xi8>,
                            %out: memref<1x12x12x256xi8>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : memref<1x14x14x256xi8>, memref<3x3xi8>)
    outs(%out : memref<1x12x12x256xi8>)
  return
}

// -----

// A stride and a bigger window are the same rewrite; only the index arithmetic
// changes. This is GoogLeNet's stem pool.
// CHECK-LABEL: func.func @a_strided_pool_is_packed
// CHECK:         memref.view {{.*}} to memref<20000xi64>
// CHECK:         scf.for {{.*}}iter_args
// CHECK-NOT:     linalg.pooling_nhwc_max
func.func @a_strided_pool_is_packed(%in: memref<1x50x50x64xi8>, %w: memref<3x3xi8>,
                                    %out: memref<1x24x24x64xi8>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%in, %w : memref<1x50x50x64xi8>, memref<3x3xi8>)
    outs(%out : memref<1x24x24x64xi8>)
  return
}

// -----

// An f32 pool has nothing to pack: four floats are a word and a float maximum
// is not a bitwise operation.
// CHECK-LABEL: func.func @an_f32_pool_is_left_alone
// CHECK:         linalg.pooling_nhwc_max
func.func @an_f32_pool_is_left_alone(%in: memref<1x14x14x256xf32>, %w: memref<3x3xf32>,
                                     %out: memref<1x12x12x256xf32>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : memref<1x14x14x256xf32>, memref<3x3xf32>)
    outs(%out : memref<1x12x12x256xf32>)
  return
}

// -----

// A channel count that is not a multiple of eight would need a remainder loop,
// and none of the models has one -- every quantized channel count here is a
// multiple of the accelerator's own 16.
// CHECK-LABEL: func.func @a_ragged_channel_count_is_left_alone
// CHECK:         linalg.pooling_nhwc_max
func.func @a_ragged_channel_count_is_left_alone(%in: memref<1x6x6x12xi8>, %w: memref<3x3xi8>,
                                                %out: memref<1x4x4x12xi8>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : memref<1x6x6x12xi8>, memref<3x3xi8>)
    outs(%out : memref<1x4x4x12xi8>)
  return
}

// -----

// Reading the bytes as words needs the buffer to start where it says: a window
// of a wider one has an offset and a stride the view cannot express.
// CHECK-LABEL: func.func @a_strided_layout_is_left_alone
// CHECK:         linalg.pooling_nhwc_max
func.func @a_strided_layout_is_left_alone(
    %in: memref<1x14x14x256xi8, strided<[200704, 14336, 1024, 1], offset: 512>>,
    %w: memref<3x3xi8>, %out: memref<1x12x12x256xi8>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : memref<1x14x14x256xi8, strided<[200704, 14336, 1024, 1], offset: 512>>, memref<3x3xi8>)
    outs(%out : memref<1x12x12x256xi8>)
  return
}

// -----

// In NCHW the channels are a whole plane apart, so eight adjacent bytes are
// eight *columns* of one channel and there is nothing to pack.
// CHECK-LABEL: func.func @nchw_is_left_alone
// CHECK:         linalg.pooling_nchw_max
func.func @nchw_is_left_alone(%in: memref<1x256x14x14xi8>, %w: memref<3x3xi8>,
                              %out: memref<1x256x12x12xi8>) {
  linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : memref<1x256x14x14xi8>, memref<3x3xi8>)
    outs(%out : memref<1x256x12x12xi8>)
  return
}

// -----

// Every value a pool reads in these models comes off a convolution with a relu,
// so it is in [0,127] -- and then bit 7 is always clear, `(a|HI) - b` cannot
// borrow out of its byte, and the whole sign dance goes away:
//
//     d   = (a | HI) - b
//     m   = ((d >> 7) & LOW) * 0xFF
//     max = b ^ ((a ^ b) & m)
//
// Eight operations instead of sixteen, and no flip on the way in or out.
// Measured on the board at GoogLeNet's shape: **-33.7%**, with the shorter
// formula checked exhaustively over all 16,384 byte pairs it claims to cover.
//
// The accumulator starts from what the output buffer holds, and a max-pool is
// filled with the i8 minimum. With a non-negative input the true maximum is
// non-negative too, so that fill becomes a zero.

// CHECK-LABEL: func.func @f_a_relu_makes_it_unsigned
// The fill is rewritten from -128 to 0:
// CHECK:         %[[Z:.*]] = arith.constant 0 : i8
// CHECK:         linalg.fill ins(%[[Z]]
// Eight operations, and no `xori` against HI on the loaded word:
// CHECK:         arith.ori
// CHECK-NEXT:    arith.subi
// CHECK-NEXT:    arith.xori
// CHECK-NEXT:    arith.shrui
// CHECK-NEXT:    arith.andi
// CHECK-NEXT:    arith.muli
// CHECK-NEXT:    arith.andi
// CHECK-NEXT:    arith.xori
// CHECK-NEXT:    scf.yield
func.func @f_a_relu_makes_it_unsigned(%x: memref<1x14x14x64xi8>, %w: memref<3x3x64x64xi8>,
                                      %win: memref<3x3xf32>, %out: memref<1x12x12x64xi8>) {
  %src = memref.alloc() : memref<1x14x14x64xi8>
  %m128 = arith.constant -128 : i8
  gemmlir.conv2d_i8(%x, %w, %src) {act = #gemmlir.act<relu>, padding = 1 : i64}
      : (memref<1x14x14x64xi8>, memref<3x3x64x64xi8>, memref<1x14x14x64xi8>)
  linalg.fill ins(%m128 : i8) outs(%out : memref<1x12x12x64xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%src, %win : memref<1x14x14x64xi8>, memref<3x3xf32>)
      outs(%out : memref<1x12x12x64xi8>)
  return
}

// -----

// Without the relu the convolution's output is a signed byte, and the sixteen
// operation form is the only correct one.

// CHECK-LABEL: func.func @g_no_relu_stays_signed
// CHECK:         arith.constant -128 : i8
// CHECK:         linalg.fill
// CHECK:         arith.xori
// CHECK:         arith.xori
func.func @g_no_relu_stays_signed(%x: memref<1x14x14x64xi8>, %w: memref<3x3x64x64xi8>,
                                  %win: memref<3x3xf32>, %out: memref<1x12x12x64xi8>) {
  %src = memref.alloc() : memref<1x14x14x64xi8>
  %m128 = arith.constant -128 : i8
  gemmlir.conv2d_i8(%x, %w, %src) {padding = 1 : i64}
      : (memref<1x14x14x64xi8>, memref<3x3x64x64xi8>, memref<1x14x14x64xi8>)
  linalg.fill ins(%m128 : i8) outs(%out : memref<1x12x12x64xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%src, %win : memref<1x14x14x64xi8>, memref<3x3xf32>)
      outs(%out : memref<1x12x12x64xi8>)
  return
}

// -----

// The padding a pool reads has to be non-negative too. `gemmlir.memset`'s value
// accessor comes back **unsigned**, so -128 reads as 128 and a plain `>= 0` is
// always true -- the byte has to be read as a signed one or this case is
// silently wrong.

// CHECK-LABEL: func.func @h_a_negative_padding_stays_signed
// CHECK:         arith.xori
// CHECK:         arith.xori
func.func @h_a_negative_padding_stays_signed(%x: memref<1x12x12x64xi8>, %w: memref<3x3x64x64xi8>,
                                             %win: memref<3x3xf32>, %out: memref<1x12x12x64xi8>) {
  %src = memref.alloc() : memref<1x14x14x64xi8>
  %m128 = arith.constant -128 : i8
  gemmlir.memset(%src) {value = -128 : i8} : memref<1x14x14x64xi8>
  %data = memref.alloc() : memref<1x12x12x64xi8>
  gemmlir.conv2d_i8(%x, %w, %data) {act = #gemmlir.act<relu>, padding = 1 : i64}
      : (memref<1x12x12x64xi8>, memref<3x3x64x64xi8>, memref<1x12x12x64xi8>)
  %inner = memref.subview %src[0, 1, 1, 0] [1, 12, 12, 64] [1, 1, 1, 1]
      : memref<1x14x14x64xi8> to memref<1x12x12x64xi8, strided<[12544, 896, 64, 1], offset: 960>>
  memref.copy %data, %inner : memref<1x12x12x64xi8> to memref<1x12x12x64xi8, strided<[12544, 896, 64, 1], offset: 960>>
  linalg.fill ins(%m128 : i8) outs(%out : memref<1x12x12x64xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%src, %win : memref<1x14x14x64xi8>, memref<3x3xf32>)
      outs(%out : memref<1x12x12x64xi8>)
  return
}

// -----

// ...and a zero padding keeps it unsigned, which is what a padded pool in
// GoogLeNet actually looks like.

// CHECK-LABEL: func.func @i_a_zero_padding_is_unsigned
// CHECK:         arith.ori
// CHECK-NEXT:    arith.subi
// CHECK-NEXT:    arith.xori
// CHECK-NEXT:    arith.shrui
func.func @i_a_zero_padding_is_unsigned(%x: memref<1x12x12x64xi8>, %w: memref<3x3x64x64xi8>,
                                        %win: memref<3x3xf32>, %out: memref<1x12x12x64xi8>) {
  %src = memref.alloc() : memref<1x14x14x64xi8>
  %m128 = arith.constant -128 : i8
  gemmlir.memset(%src) {value = 0 : i8} : memref<1x14x14x64xi8>
  %data = memref.alloc() : memref<1x12x12x64xi8>
  gemmlir.conv2d_i8(%x, %w, %data) {act = #gemmlir.act<relu>, padding = 1 : i64}
      : (memref<1x12x12x64xi8>, memref<3x3x64x64xi8>, memref<1x12x12x64xi8>)
  %inner = memref.subview %src[0, 1, 1, 0] [1, 12, 12, 64] [1, 1, 1, 1]
      : memref<1x14x14x64xi8> to memref<1x12x12x64xi8, strided<[12544, 896, 64, 1], offset: 960>>
  memref.copy %data, %inner : memref<1x12x12x64xi8> to memref<1x12x12x64xi8, strided<[12544, 896, 64, 1], offset: 960>>
  linalg.fill ins(%m128 : i8) outs(%out : memref<1x12x12x64xi8>)
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%src, %win : memref<1x14x14x64xi8>, memref<3x3xf32>)
      outs(%out : memref<1x12x12x64xi8>)
  return
}

// -----

// A **band** -- what `--pool-without-padding` writes -- is a rank-preserving,
// unit-stride subview that cuts only the spatial dimensions. The bytes are
// still eight to a word; the band starts further into the buffer. Refusing
// these is what kept the banding away from every pool this pass can pack, and
// with it the padded copy the banding exists to remove.

// CHECK-LABEL: func.func @a_band_packs
// CHECK:         memref.view %{{.*}} : memref<32768xi8> to memref<4096xi64>
// CHECK:         memref.view %{{.*}} : memref<8192xi8> to memref<1024xi64>
// CHECK:         scf.for
// CHECK-NOT:     linalg.pooling_nhwc_max
func.func @a_band_packs(%src: memref<1x16x16x128xi8>, %out: memref<1x8x8x128xi8>,
                        %win: memref<3x3xf32>) {
  %lo = arith.constant 0 : i8
  %si = memref.subview %src[0, 1, 1, 0] [1, 15, 15, 128] [1, 1, 1, 1]
    : memref<1x16x16x128xi8> to memref<1x15x15x128xi8, strided<[32768, 2048, 128, 1], offset: 2176>>
  %so = memref.subview %out[0, 1, 1, 0] [1, 7, 7, 128] [1, 1, 1, 1]
    : memref<1x8x8x128xi8> to memref<1x7x7x128xi8, strided<[8192, 1024, 128, 1], offset: 1152>>
  linalg.fill ins(%lo : i8) outs(%so : memref<1x7x7x128xi8, strided<[8192, 1024, 128, 1], offset: 1152>>)
  linalg.pooling_nhwc_max {strides = dense<2> : vector<2xi64>,
                           dilations = dense<1> : vector<2xi64>}
    ins(%si, %win : memref<1x15x15x128xi8, strided<[32768, 2048, 128, 1], offset: 2176>>, memref<3x3xf32>)
    outs(%so : memref<1x7x7x128xi8, strided<[8192, 1024, 128, 1], offset: 1152>>)
  return
}

// -----

// A subview that cuts the **channel** axis is not a band: the packing is made
// of eight consecutive channels, and half of them would be another slice's.

// CHECK-LABEL: func.func @a_channel_slice_is_not_a_band
// CHECK:         linalg.pooling_nhwc_max
// CHECK-NOT:     memref.view
func.func @a_channel_slice_is_not_a_band(%src: memref<1x16x16x128xi8>,
                                         %out: memref<1x8x8x128xi8>,
                                         %win: memref<3x3xf32>) {
  %lo = arith.constant 0 : i8
  %si = memref.subview %src[0, 0, 0, 0] [1, 15, 15, 64] [1, 1, 1, 1]
    : memref<1x16x16x128xi8> to memref<1x15x15x64xi8, strided<[32768, 2048, 128, 1]>>
  %so = memref.subview %out[0, 0, 0, 0] [1, 7, 7, 64] [1, 1, 1, 1]
    : memref<1x8x8x128xi8> to memref<1x7x7x64xi8, strided<[8192, 1024, 128, 1]>>
  linalg.fill ins(%lo : i8) outs(%so : memref<1x7x7x64xi8, strided<[8192, 1024, 128, 1]>>)
  linalg.pooling_nhwc_max {strides = dense<2> : vector<2xi64>,
                           dilations = dense<1> : vector<2xi64>}
    ins(%si, %win : memref<1x15x15x64xi8, strided<[32768, 2048, 128, 1]>>, memref<3x3xf32>)
    outs(%so : memref<1x7x7x64xi8, strided<[8192, 1024, 128, 1]>>)
  return
}
