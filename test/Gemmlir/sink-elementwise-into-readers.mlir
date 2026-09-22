// RUN: gemmlir-opt --sink-elementwise-into-readers --split-input-file %s | FileCheck %s

// A ViT's layer norm writes `x - mean` into a 17x192 f32 buffer -- 13 KB
// against a 16 KB L1 -- and reads it twice, once for the variance and once to
// normalize. Upstream elementwise fusion refuses it: fusing wants `hasOneUse`
// and this has two. Recomputing the subtraction in both readers is -12.8% on
// the layer norm, measured as a kernel on the board, and byte for byte the same
// answer -- the same arithmetic in the same order, only not stored in between.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @a_layer_norm_centres_twice
// CHECK:       %[[X:.*]] = memref.alloc() : memref<17x192xf32>
// CHECK:       %[[M:.*]] = memref.alloc() : memref<17xf32>
// CHECK:       %[[R:.*]] = memref.alloc() : memref<17xf32>
// CHECK:       %[[V:.*]] = memref.alloc() : memref<17xf32>
// CHECK:       %[[O:.*]] = memref.alloc() : memref<17x192xf32>
// CHECK-NOT:   memref.alloc
// CHECK:       linalg.generic
// CHECK-SAME:  ins(%[[X]], %[[M]] :
// CHECK-SAME:  outs(%[[V]]
// CHECK:         arith.subf
// CHECK:         math.fma
// CHECK:       linalg.generic
// CHECK-SAME:  ins(%[[X]], %[[M]], %[[R]] :
// CHECK-SAME:  outs(%[[O]]
// CHECK:         arith.subf
// CHECK:         arith.mulf
// CHECK-NOT:   linalg.generic
func.func @a_layer_norm_centres_twice() {
  %x = memref.alloc() : memref<17x192xf32>
  %mean = memref.alloc() : memref<17xf32>
  %r = memref.alloc() : memref<17xf32>
  %var = memref.alloc() : memref<17xf32>
  %out = memref.alloc() : memref<17x192xf32>
  %c = memref.alloc() : memref<17x192xf32>
  linalg.generic {indexing_maps = [#id2, #row, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x, %mean : memref<17x192xf32>, memref<17xf32>) outs(%c : memref<17x192xf32>) {
  ^bb0(%in: f32, %m: f32, %o: f32):
    %s = arith.subf %in, %m : f32
    linalg.yield %s : f32
  }
  linalg.generic {indexing_maps = [#id2, #row], iterator_types = ["parallel", "reduction"]}
      ins(%c : memref<17x192xf32>) outs(%var : memref<17xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %v = math.fma %in, %in, %acc : f32
    linalg.yield %v : f32
  }
  linalg.generic {indexing_maps = [#id2, #row, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%c, %r : memref<17x192xf32>, memref<17xf32>) outs(%out : memref<17x192xf32>) {
  ^bb0(%in: f32, %ri: f32, %o: f32):
    %n = arith.mulf %in, %ri : f32
    linalg.yield %n : f32
  }
  memref.dealloc %c : memref<17x192xf32>
  return
}

// -----

// The bound is traffic, not count. Two inputs the size of the buffer and three
// readers is `2*2B` against `4*B`: recomputing moves more bytes than it saves,
// and DenseNet's dequantize with up to 25 readers measured +0.9% on the board.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @b_three_readers_of_two_inputs
// CHECK-COUNT-4: linalg.generic
func.func @b_three_readers_of_two_inputs() {
  %a = memref.alloc() : memref<17x192xf32>
  %b = memref.alloc() : memref<17x192xf32>
  %o1 = memref.alloc() : memref<17x192xf32>
  %o2 = memref.alloc() : memref<17x192xf32>
  %o3 = memref.alloc() : memref<17x192xf32>
  %s = memref.alloc() : memref<17x192xf32>
  linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<17x192xf32>, memref<17x192xf32>) outs(%s : memref<17x192xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.addf %x, %y : f32
    linalg.yield %v : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o1 : memref<17x192xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o2 : memref<17x192xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o3 : memref<17x192xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  }
  memref.dealloc %s : memref<17x192xf32>
  return
}

// -----

// A dequantize is a quarter of the buffer it writes, so even three readers pay
// -- that half of the rule has to be there too.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @c_a_narrow_input_pays
// CHECK-COUNT-2: arith.sitofp
// CHECK-NOT:   arith.sitofp
func.func @c_a_narrow_input_pays() {
  %cst = arith.constant 3.000000e-02 : f32
  %q = memref.alloc() : memref<17x192xi8>
  %o1 = memref.alloc() : memref<17x192xf32>
  %o2 = memref.alloc() : memref<17x192xf32>
  %s = memref.alloc() : memref<17x192xf32>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%q : memref<17x192xi8>) outs(%s : memref<17x192xf32>) {
  ^bb0(%x: i8, %o: f32):
    %f = arith.sitofp %x : i8 to f32
    %v = arith.mulf %f, %cst : f32
    linalg.yield %v : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o1 : memref<17x192xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o2 : memref<17x192xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  }
  memref.dealloc %s : memref<17x192xf32>
  return
}

// -----

// Recomputing reads the producer's inputs where the reader stands, so a write
// to one of them in between stops it. The `linalg.fill` here is the whole
// difference.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @d_an_input_written_in_between
// CHECK-COUNT-3: linalg.generic
func.func @d_an_input_written_in_between() {
  %cst = arith.constant 1.000000e+00 : f32
  %o1 = memref.alloc() : memref<17x192xf32>
  %o2 = memref.alloc() : memref<17x192xf32>
  %x = memref.alloc() : memref<17x192xf32>
  %s = memref.alloc() : memref<17x192xf32>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x : memref<17x192xf32>) outs(%s : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    %v = arith.mulf %a, %cst : f32
    linalg.yield %v : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o1 : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  linalg.fill ins(%cst : f32) outs(%x : memref<17x192xf32>)
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o2 : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  memref.dealloc %s : memref<17x192xf32>
  memref.dealloc %x : memref<17x192xf32>
  return
}

// -----

// A `memref.subview` of the buffer is a second name for it, and a reader
// reached through one is not a reader this pass can rewrite.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @e_an_alias_of_the_buffer
// CHECK:       memref.alloc
// CHECK:       arith.mulf
func.func @e_an_alias_of_the_buffer() -> memref<1x192xf32, strided<[192, 1]>> {
  %cst = arith.constant 2.000000e+00 : f32
  %x = memref.alloc() : memref<17x192xf32>
  %o1 = memref.alloc() : memref<17x192xf32>
  %s = memref.alloc() : memref<17x192xf32>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x : memref<17x192xf32>) outs(%s : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    %v = arith.mulf %a, %cst : f32
    linalg.yield %v : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o1 : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  %v = memref.subview %s[0, 0] [1, 192] [1, 1] : memref<17x192xf32> to memref<1x192xf32, strided<[192, 1]>>
  return %v : memref<1x192xf32, strided<[192, 1]>>
}

// -----

// `linalg.index` reads the loop it sits in, and the reader's loop is a
// different one.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @f_a_body_that_reads_its_index
// CHECK:       memref.alloc
// CHECK:       linalg.index
func.func @f_a_body_that_reads_its_index() {
  %x = memref.alloc() : memref<17x192xf32>
  %o1 = memref.alloc() : memref<17x192xf32>
  %o2 = memref.alloc() : memref<17x192xf32>
  %s = memref.alloc() : memref<17x192xf32>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x : memref<17x192xf32>) outs(%s : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    %i = linalg.index 1 : index
    %n = arith.index_cast %i : index to i32
    %f = arith.sitofp %n : i32 to f32
    %v = arith.addf %a, %f : f32
    linalg.yield %v : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o1 : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%s : memref<17x192xf32>) outs(%o2 : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  memref.dealloc %s : memref<17x192xf32>
  return
}

// -----

// A reader can take a slice of the buffer rather than the whole of it, and then
// recomputing costs nothing at all: a transformer's Q, K and V are three
// **disjoint** thirds of one dequantized projection, so the three of them
// together do exactly the work the one producer did. The reindexing is replayed
// on the producer's input, which is i32 where the buffer was f32.
//
// `vit_tiny` has twelve of these and they are its largest single block of
// elementwise work.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#swap = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @g_three_disjoint_slices
// CHECK-NOT:   memref<17x576xf32>
// CHECK:       memref.expand_shape %{{.*}} : memref<17x576xi32> into memref<1x17x576xi32>
// CHECK-COUNT-3: arith.sitofp
// CHECK-NOT:   arith.sitofp
func.func @g_three_disjoint_slices() {
  %cst = arith.constant 3.000000e-02 : f32
  %src = memref.alloc() : memref<17x576xi32>
  %buf = memref.alloc() : memref<17x576xf32>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%src : memref<17x576xi32>) outs(%buf : memref<17x576xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %v = arith.mulf %f, %cst : f32
    linalg.yield %v : f32
  }
  %e = memref.expand_shape %buf [[0, 1], [2]] output_shape [1, 17, 576]
      : memref<17x576xf32> into memref<1x17x576xf32>
  %q = memref.subview %e[0, 0, 0] [1, 17, 192] [1, 1, 1]
      : memref<1x17x576xf32> to memref<1x17x192xf32, strided<[9792, 576, 1]>>
  %k = memref.subview %e[0, 0, 192] [1, 17, 192] [1, 1, 1]
      : memref<1x17x576xf32> to memref<1x17x192xf32, strided<[9792, 576, 1], offset: 192>>
  %v = memref.subview %e[0, 0, 384] [1, 17, 192] [1, 1, 1]
      : memref<1x17x576xf32> to memref<1x17x192xf32, strided<[9792, 576, 1], offset: 384>>
  %qe = memref.expand_shape %q [[0], [1], [2, 3]] output_shape [1, 17, 3, 64]
      : memref<1x17x192xf32, strided<[9792, 576, 1]>> into memref<1x17x3x64xf32, strided<[9792, 576, 64, 1]>>
  %ke = memref.expand_shape %k [[0], [1], [2, 3]] output_shape [1, 17, 3, 64]
      : memref<1x17x192xf32, strided<[9792, 576, 1], offset: 192>> into memref<1x17x3x64xf32, strided<[9792, 576, 64, 1], offset: 192>>
  %ve = memref.expand_shape %v [[0], [1], [2, 3]] output_shape [1, 17, 3, 64]
      : memref<1x17x192xf32, strided<[9792, 576, 1], offset: 384>> into memref<1x17x3x64xf32, strided<[9792, 576, 64, 1], offset: 384>>
  %oq = memref.alloc() : memref<1x3x17x64xf32>
  %ok = memref.alloc() : memref<1x3x17x64xf32>
  %ov = memref.alloc() : memref<1x3x17x64xf32>
  linalg.generic {indexing_maps = [#swap, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%qe : memref<1x17x3x64xf32, strided<[9792, 576, 64, 1]>>) outs(%oq : memref<1x3x17x64xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  linalg.generic {indexing_maps = [#swap, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ke : memref<1x17x3x64xf32, strided<[9792, 576, 64, 1], offset: 192>>) outs(%ok : memref<1x3x17x64xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  linalg.generic {indexing_maps = [#swap, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ve : memref<1x17x3x64xf32, strided<[9792, 576, 64, 1], offset: 384>>) outs(%ov : memref<1x3x17x64xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  memref.dealloc %buf : memref<17x576xf32>
  return
}

// -----

// `scf.for` declares no memory effects of its own, so asking it directly says
// "unknown" and refuses every transformer -- a reader past an accelerator call
// in a loop is exactly where this shows up. The walk goes inside it instead,
// and a `gemmlir` operation's write is tied to no operand, so every buffer it
// holds counts as the target.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @h_a_reader_past_a_loop
// CHECK-COUNT-2: arith.sitofp
// CHECK-NOT:   arith.sitofp
func.func @h_a_reader_past_a_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.000000e-02 : f32
  %src = memref.alloc() : memref<17x192xi32>
  %buf = memref.alloc() : memref<17x192xf32>
  %lhs = memref.alloc() : memref<17x192xi8>
  %w = memref.alloc() : memref<192x64xi8>
  %acc = memref.alloc() : memref<17x64xi32>
  %o1 = memref.alloc() : memref<17x192xf32>
  %o2 = memref.alloc() : memref<17x192xf32>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%src : memref<17x192xi32>) outs(%buf : memref<17x192xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %v = arith.mulf %f, %cst : f32
    linalg.yield %v : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%buf : memref<17x192xf32>) outs(%o1 : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  scf.for %i = %c0 to %c3 step %c1 {
    gemmlir.matmul_i8(%lhs, %w, %acc) : (memref<17x192xi8> x memref<192x64xi8>) -> memref<17x64xi32> {accumulate = false}
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%buf : memref<17x192xf32>) outs(%o2 : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  memref.dealloc %buf : memref<17x192xf32>
  return
}

// -----

// ...and the same loop writing something the producer reads stops it, which is
// the check that makes the one above sound rather than optimistic.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @i_a_loop_that_writes_the_input
// CHECK:       arith.sitofp
// CHECK-NOT:   arith.sitofp
func.func @i_a_loop_that_writes_the_input() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.000000e-02 : f32
  %src = memref.alloc() : memref<17x192xi32>
  %buf = memref.alloc() : memref<17x192xf32>
  %lhs = memref.alloc() : memref<17x192xi8>
  %w = memref.alloc() : memref<192x192xi8>
  %o1 = memref.alloc() : memref<17x192xf32>
  %o2 = memref.alloc() : memref<17x192xf32>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%src : memref<17x192xi32>) outs(%buf : memref<17x192xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %v = arith.mulf %f, %cst : f32
    linalg.yield %v : f32
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%buf : memref<17x192xf32>) outs(%o1 : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  scf.for %i = %c0 to %c3 step %c1 {
    gemmlir.matmul_i8(%lhs, %w, %src) : (memref<17x192xi8> x memref<192x192xi8>) -> memref<17x192xi32> {accumulate = false}
  }
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%buf : memref<17x192xf32>) outs(%o2 : memref<17x192xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  }
  memref.dealloc %buf : memref<17x192xf32>
  return
}
