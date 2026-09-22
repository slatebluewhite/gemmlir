//===- GemmlirPasses.td - Gemmlir dialect passes -------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef GEMMLIR_PASS
#define GEMMLIR_PASS

include "mlir/Pass/PassBase.td"

// Linalg -> Gemmlir conversion pass (matmul lowering).
def ConvertLinalgToGemmlir: Pass<"convert-linalg-to-gemmlir", "::mlir::ModuleOp"> {
  let summary = "Lower linalg.matmul to gemmlir.matmul ops";
  let description = [{
    Rewrites linalg.matmul operations into gemmlir dialect matmul ops
    (matmul_i8 or matmul_i8_scale) depending on output element type.

    The `dataflow` option picks the Gemmini dataflow recorded on the produced
    ops, which `--convert-gemmlir-to-llvm` passes to the runtime as
    tiled_matmul_type. Whether `os` is usable depends on the bitstream.
  }];

  let options = [
    Option<"dataflow", "dataflow", "std::string", /*default=*/"\"ws\"",
           "Gemmini dataflow to request: os, ws or cpu">,
    Option<"fusePooling", "fuse-pooling", "bool", /*default=*/"false",
           "Fold a max-pool into the convolution call (off: see docs/pipeline.md)">
  ];
}

// Gemmlir -> LLVM conversion pass.
def ConvertGemmlirToLLVM: Pass<"convert-gemmlir-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower gemmlir ops to LLVM dialect";
  let description = [{
    Lowers gemmlir.matmul_i8 into a call to an external C runtime
    function (e.g. tiled_matmul_auto) and relies on standard conversions
    to lower remaining types/ops to the LLVM dialect.
  }];
}

// Force F32 matmul into INT8xINT8->INT32 with q/dq around it.
def ForceQuantizedMatmul: Pass<"force-quantized-matmul", "::mlir::ModuleOp"> {
  let summary = "Force-quantize linalg.matmul to int8 path";
  let description = [{
    Converts linalg.matmul with f32 operands/results into a quantized pattern
    using int8 x int8 -> int32 accumulation, inserting corresponding quant
    and dequant operations.

    Scales are symmetric and per tensor: `max|x| / 127`. A constant operand --
    the weights -- is measured here; an activation cannot be, so a calibration
    step has to supply it. Put it on the operation as
    `gemmlir.activation_scale = <f64>`; `activation-scale` is the fallback for
    operations that carry no annotation. The accumulator's scale is the product
    of the two.

    One scale for a whole network is not enough: on a two-layer MLP the two
    matmuls saw activations reaching 3.74 and 2.11, and sharing a scale left a
    relative L2 error of 0.20 against PyTorch where per-operation scales did
    better.
  }];

  let options = [
    Option<"activationScale", "activation-scale", "double", /*default=*/"2.0e-2",
           "Symmetric int8 scale for operands whose range is not known at "
           "compile time; max|x|/127 over representative inputs">
  ];
}

// Works around the bare-pointer calling convention returning the allocated
// rather than the aligned pointer of a memref result.
def LegalizeBarePtrReturns: Pass<"legalize-bare-ptr-returns", "::mlir::func::FuncOp"> {
  let summary = "Make a returned memref's allocated and aligned pointers coincide";
  let description = [{
    Under `--convert-func-to-llvm=use-bare-ptr-memref-call-conv=1` a function
    returning a memref hands back one pointer, and that pointer is the
    *allocated* one. When the memref came from a `memref.alloc` carrying an
    `alignment`, the data starts up to `alignment - 1` bytes later, so the caller
    reads the wrong memory -- silently, since nothing in the pipeline notices.

    This pass drops the `alignment` attribute from allocations that are returned
    directly, which makes both pointers the malloc result. Run it before
    `--convert-func-to-llvm`. It is a no-op for functions that return nothing or
    return memrefs they did not allocate.
  }];
}

// The quant dialect's lowering truncates where quantization should round.
def ConvNchwToNhwc: Pass<"conv-nchw-to-nhwc", "::mlir::func::FuncOp"> {
  let summary = "Rewrite NCHW convolution and pooling into their NHWC forms";
  let description = [{
    PyTorch hands over `linalg.conv_2d_nchw_fchw`, and the layout decides how
    much of the layer can be offloaded. From NHWC, MLIR's own img2col contracts
    `X(P x CKhKw) * W(CKhKw x F)` -- channels in the columns -- which is the one
    form Gemmini's `repeating_bias` can take, so the bias, the requantization
    and the activation all fold into the accelerator call. From NCHW the
    contraction comes out the other way round, the bias is per row, and the
    whole tail of the layer stays in software.

    The convolution, the pooling and their operands get explicit
    `linalg.transpose`s; `--canonicalize` then composes the ones that meet and
    drops the identities, and this pass pushes a transpose past the elementwise
    operations between two layers so that they do meet. What survives is one
    transpose on the input and one on the way out to the classifier.

    Run it before `--conv-to-img2col`.
  }];
}

def PlanStaticBuffers: Pass<"plan-static-buffers", "::mlir::ModuleOp"> {
  let summary = "Lay the function's temporaries out in one static arena";
  let description = [{
    Two things want this.

    Gemmini returns wrong data on this board when the buffer it writes moves
    between calls -- see docs/pipeline.md. Bufferization allocates every
    intermediate with `memref.alloc`, so whether a model is correct depends on
    whether the allocator happens to hand back the same addresses each time; the
    CNN was stable until folding a pooling away removed one allocation and made
    another alternate between two bins.

    And the allocation itself is not free. Every call to an inference function
    that allocates and frees is a fresh mapping the kernel has to populate --
    with `mlockall(MCL_FUTURE)`, at `mmap` time -- so a MobileNetV2 whose
    temporaries come to 4.45 MB spent **56 ms an inference**, 45% of its run
    time, on nothing but that.

    So the temporaries become views into one uninitialized `memref.global`, at
    offsets assigned by first fit over their live ranges. Buffers whose lives do
    not overlap share space, and the arena is the peak rather than the sum. The
    function is then not re-entrant, which is what an inference entry point
    normally is anyway; a buffer the function returns, or one reachable from a
    returned view, keeps the allocator's lifetime, since its ownership is the
    caller's.

    Live ranges are positions along one line of operations, so a function whose
    control flow has been flattened into several blocks -- a batched model, from
    the loop over its images -- gets no sharing and the arena is the sum. That
    costs address space and nothing else. Allocations inside a region are left
    alone entirely.

    Run it after `--convert-linalg-to-gemmlir`.
  }];
}

def QuantizeUnfoldableTails: Pass<"quantize-unfoldable-tails", "::mlir::func::FuncOp"> {
  let summary = "Requantize a layer whose activation the accelerator cannot end in";
  let description = [{
    `conv2d_i8` and `depthwise_conv2d_i8` write i8, so the tail has to be
    `saturate(scale * accumulator + bias)` with at most a relu. EfficientNet's
    SiLU is none of those, and a depthwise convolution has no matmul to fall
    back on the way a dense one does -- sixteen of them stayed scalar loops.

    Quantizing the layer's own output first splits the tail into the part the
    call can do and the part the core does afterwards. On the board at 64x64
    those sixteen convolutions are **1749 ms of EfficientNet's 2982**, and the
    extra round trip costs **0.00016** relative L2 against a model whose own
    quantization error is 0.0186.

    Needs `gemmlir.output_scale`, which the calibration measures separately from
    the next layer's input range -- there is an activation between them.
  }];
}

def QuantizeBiasIntoAccumulator: Pass<"quantize-bias-into-accumulator", "::mlir::func::FuncOp"> {
  let summary = "Add a constant bias to the integer accumulator instead of after dequantizing";
  let description = [{
    A quantized layer comes out of a frontend adding its bias in f32, after the
    accumulator has been scaled back: `acc * s + b`. The accelerator cannot do
    that -- its bias operand `D` is added to the i32 accumulator *before* the
    mvout scaling -- so the bias keeps the whole tail of the layer in software.

    Since `acc * s + b == (acc + b/s) * s`, a constant bias can move inside as
    `round(b/s)`, evaluated at compile time. That is the standard int8 bias
    quantization and the error is half a step of `s`, which is the accumulator's
    resolution and far below the output's: on the CNN's first convolution
    `s / s_out` is 0.0029, so the bias is exact to 0.0015 of an output LSB.

    Run it after the elementwise fusion has put the scaling and the bias add in
    one operation, and before bufferization. `--convert-linalg-to-gemmlir` then
    folds the integer add into the runtime's `D` operand.
  }];
}

def ExpandStaticMemrefCopy: Pass<"expand-static-memref-copy", "::mlir::func::FuncOp"> {
  let summary = "Write a statically shaped memref.copy out as a loop nest";
  let description = [{
    The runtime's copy reads a descriptor, decides how much of the shape is
    packed on both sides and whether the run is word-aligned, then runs an
    odometer -- the same answer every time for a given call site. Written out
    here the run is a constant, so its loads and stores are unrolled and the
    index arithmetic is two `addi`: **36 -> 19.7 cycles a run** on `gmid`'s
    im2col pack.

    Only when it can be proved: a run between 16 and 64 bytes and a whole
    number of words, both bases and offsets eight-byte aligned, and every
    remaining stride a multiple of eight.
  }];
}

def FoldRelayoutIntoProducers: Pass<"fold-relayout-into-producers", "::mlir::func::FuncOp"> {
  let summary = "Have the operations that filled a buffer write its relayout instead";
  let description = [{
    A grouped convolution's four tails each dequantize eight channels into a
    thirty-two channel NHWC buffer, which is then relaid out to the NCHW the
    model returns. Each tail already walks its own iteration space; writing the
    permuted slice instead costs it nothing and the relayout goes away.

    A model's worth of that work: four tails then a relayout is 6.20 ms, four
    permuting tails 2.63 -- which is the whole of why `gmin` costs 25% more than
    `gmid`, the same convolution without one.

    The slices have to tile the buffer along one axis and cover it.
  }];
}

def FuseMultiplyAdd: Pass<"fuse-multiply-add", "::mlir::func::FuncOp"> {
  let summary = "Rewrite a multiply feeding an add as math.fma";
  let description = [{
    `fmadd.s` is one trip through the FPU where `fmul.s` then `fadd.s` is two,
    and on this in-order core the add waits for the multiply. The dequantize
    tail of every unfoldable convolution is exactly that shape.

    Run **after** `--convert-linalg-to-gemmlir`, which reads the `mulf`/`addf`
    pair to recognise a dequantization it can fold into the accelerator.

    Not bit-exact: the fused form rounds once instead of twice, which is more
    accurate, not less.
  }];
}

def TableForInt8Elementwise: Pass<"table-for-i8-elementwise", "::mlir::ModuleOp"> {
  let summary = "An elementwise function of one i8 has 256 answers; look them up";
  let description = [{
    What a requantization in front of an activation buys beyond the offload it
    was put there for. EfficientNet's SiLU is `x * sigmoid(x)` -- seven
    operations including a `math.exp` at about 65 cycles -- and it reads the i8
    `--quantize-unfoldable-tails` leaves behind, so the whole chain is a lookup
    in a 256-entry table.

    Exact by construction: each entry is what the loop computes for that input,
    with `expf` and not `exp`, because that is what the loop would have called.
  }];
}

def PromoteReductionAccumulator: Pass<"promote-reduction-accumulator", "::mlir::func::FuncOp"> {
  let summary = "Carry a reduction's running value in a register, not a memref slot";
  let description = [{
    `--convert-linalg-to-loops` writes a reduction's running value back to its
    output buffer on every step and reads it again on the next one. A 3x3
    max-pool pays nine store-to-load round trips per output where one store
    would do, and on this in-order core that round trip *is* the loop --
    sampling GoogLeNet's program counter puts **82% of the whole model** in
    loops of exactly this shape.

    Nothing downstream removes it: `llc` runs no IR pipeline at all, so there is
    no pass to promote the slot.

    Rewriting it as an `scf.for` iteration argument is the same computation with
    the memory taken out of the middle, and it cascades -- promoting the
    innermost loop leaves the load and the store in the loop above, which is
    then a candidate itself, so a window ends with one load before the nest and
    one store after it.

    It asks for a slot nothing else can touch: one store in the loop, one load
    of the same slot, no other use of that buffer anywhere inside, no call, and
    nothing else that writes memory at any depth. A store under an `scf.if`
    updates the slot only sometimes, so both have to sit directly in the loop's
    own body.
  }];
}

def CombineChannelAffine: Pass<"combine-channel-affine", "::mlir::func::FuncOp"> {
  let summary = "One multiply-add on two per-channel numbers, not three operations an element";
  let description = [{
    A batch norm that could not fold into anybody's weights spends three
    operations an element on numbers that only change per channel:
    `((x - mean[c]) * rsqrt[c] + beta) / scale`. DenseNet-121 has 593 of them --
    every dense layer renormalizes the whole concatenated stack, and the
    parameters belong to the *consumer*, so there are no weights to fold into.
    `fsub.s` and `fmul.s` alone are 11.6% of that model by program-counter
    sampling, and they are two of the five steps on the dependency chain.

    `((x - m) * r + b) / s` is `x * (r/s) + (b - m*r)/s`: one multiply-add on two
    per-channel numbers, computed once per channel in their own loop. The scale
    moves inside the activation, which is sound because `max(y, 0)/s` is
    `max(y/s, 0)` for a positive `s`, and on a NaN or a negative zero both forms
    agree.

    The two forms differ by rounding, so the rewrite is confined to a value that
    reaches a conversion to an integer and nothing else -- the same condition
    and the same argument as `--hoist-invariant-reciprocal`. Run it after that
    pass, which is what turns the `rsqrt` into a per-channel buffer, and before
    `--plan-static-buffers`, which is what stops the two new buffers being a
    malloc per inference.
  }];
}

def PackInt8MaxPool: Pass<"pack-int8-max-pool", "::mlir::func::FuncOp"> {
  let summary = "Eight channels of an i8 max-pool in one 64-bit word";
  let description = [{
    A max-pool over i8 is independent per channel and NHWC puts the channels next
    to each other, so eight of them are one 64-bit word and the whole window can
    be walked eight at a time.

    The scalar form is what a program-counter profile finds at the top of
    GoogLeNet: 39% of the model in nine byte loads and eight compares per output,
    each compare a data-dependent branch. Measured on the board on one of its
    shapes, 200 repetitions: 111.10 ms scalar against 21.65 ms packed, **5.13x**.

    The byte-wise unsigned maximum is checked exhaustively -- all 65,536 byte
    pairs and 200,000 random word pairs -- in `scripts/poolswar.c`; the obvious
    shorter formula is wrong on 248,893 of them.

    Needs a dense buffer with an identity layout, a channel count divisible by
    eight, and NHWC: in NCHW the channels are a plane apart and there is nothing
    to pack.
  }];
}

def ProbeBuffers: Pass<"probe-buffers", "::mlir::func::FuncOp"> {
  let summary = "A checksum call after every buffer write (debugging only)";
  let description = [{
    Puts a call to `gemmlir_probe(pointer, bytes, id)` after every operation that
    writes a buffer, so two builds that should agree can be compared one write at
    a time instead of only at the model's output.

    **Never in the pipeline.** It is here because the check that has caught
    everything else -- the same object against the accelerator runtime and
    against the CPU one -- cannot see a difference in the *generated host code*,
    since both runtimes share it. Comparing two builds byte for byte says that
    they differ; this says where.

    The probe covers the whole allocation the write lands in, not the view: a
    padded buffer written through a subview is exactly the case worth watching,
    and its border is part of what the next operation reads.
  }];
}

def SaturateConstantCasts: Pass<"saturate-constant-casts", "::mlir::func::FuncOp"> {
  let summary = "Fold a float constant's conversion to an integer, saturating";
  let description = [{
    A padded max-pool arrives from the frontend with `-inf` in its border, and
    when the pool moves onto i8 the quantization of that constant is left in
    front of it: `trunci(minsi(maxsi(fptosi(-inf), -128), 127))`.

    `arith.fptosi` of a value outside the destination's range is **poison**, and
    `maxsi(poison, -128)` is poison too, so the clamp does not rescue it. What
    the border holds is then whatever the backend materializes, which changes
    with the surrounding code: GoogLeNet's answer moved by 0.6% of its output
    range when `--unroll-elementwise-loops` went from two to four, and this one
    byte was why.

    Replacing poison with a defined value is a refinement, so folding is always
    legal. Folding it the way the hardware does -- saturate to the extremes, and
    the maximum for a NaN, which is what RISC-V's `fcvt.w.s` gives -- makes the
    compiled answer and the obvious reading of the source agree. For a max-pool
    that is `INT_MIN`, which the clamp turns into the i8 minimum: the pool's own
    identity, which is what the padding always meant.
  }];
}

def DropClampBelowRelu: Pass<"drop-clamp-below-relu", "::mlir::func::FuncOp"> {
  let summary = "A quantizer below an activation does not need its lower clamp";
  let description = [{
    A quantization tail is `clamp(round(x / s), -128, 127)`, and on a core with
    no `Zbb` each end of that clamp is a compare, a branch and a `li`. Where the
    value came through a relu it is already at least zero, so the lower end can
    never bite: `arith.maxnumf(x, 0)` hands back the operand that is not a NaN,
    so its result is non-negative whatever `x` was, and neither
    `math.roundeven` nor a multiply by a positive scale can take it below zero.

    This is not the rewrite measured and refused twice -- that one moved the
    clamp into the float, putting two more operations *on* the dependency chain.
    This takes two instructions away and adds nothing.

    An out-of-range value makes `arith.fptosi` poison and `maxsi(poison, -128)`
    is poison too, so dropping the maximum leaves the poison where it was.
  }];
}

def SetTargetDataLayout: Pass<"set-target-data-layout", "::mlir::ModuleOp"> {
  let summary = "Tell the translation which machine this is for";
  let description = [{
    Stamps `llvm.data_layout` and `llvm.target_triple` on the module.

    Nothing set them, so `mlir-translate` handed `llc` a module with no
    `target datalayout` line -- and LLVM's default layout says an `i64` is
    aligned to **four** bytes. Every load and store of one is then printed
    `align 4`, and the RISC-V backend, which must honour that, splits it into
    two 32-bit accesses plus a shift and an or. `--pack-int8-max-pool` exists to
    put eight channels in one `i64`, so it was paying that on every load of
    every window.

    Only `mlir-translate` reads this attribute: `DataLayoutAnalysis`, which is
    what the conversions consult, looks at `dlti.dl_spec`. So the pass changes
    nothing before translation and can sit at the end of the pipeline.

    The defaults are this board's: rv64gc with the lp64d ABI.
  }];
  let options = [
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"e-m:e-p:64:64-i64:64-i128:128-n32:64-S128\"",
           "LLVM target data layout string; empty leaves the module alone">,
    Option<"targetTriple", "target-triple", "std::string",
           /*default=*/"\"riscv64-unknown-linux-gnu\"",
           "LLVM target triple; empty leaves the module alone">,
  ];
}

def FillOnlyTheBorder: Pass<"fill-only-the-border", "::mlir::func::FuncOp"> {
  let summary = "A padding is the border, not the whole buffer";
  let description = [{
    A padding fills the whole buffer and then the real data is written into the
    middle of it, so most of what was just written is overwritten before
    anything reads it. Across GoogLeNet's ten padded pools that is 681,808
    elements filled of which 473,472 -- 69% -- never survive, and the fill is a
    scalar store loop rather than a `memset`, because the buffer is read by a
    pool and `--fill-to-memset` refuses a fill a reduction touches.

    The complement of a box inside a buffer is a set of slabs, one pair per
    dimension: a point outside the box is outside on some dimension, and
    assigning it to the first such dimension -- dimensions before it clipped to
    the box, dimensions after it left whole -- covers every outside point
    exactly once. For an NHWC pad of `high[0, 2, 2, 0]` that is the last two
    rows entire, then the last two columns of the rows above them.

    Sound because the fill and the write cover the buffer between them either
    way: every element the removed part of the fill wrote is written again by
    the very next operation to touch the buffer, before any read.
  }];
  let options = [
    Option<"minSaved", "min-saved", "int64_t", /*default=*/"256",
           "Skip a site that would spare fewer elements than this">,
  ];
}

def PoolWithoutPadding: Pass<"pool-without-padding", "::mlir::func::FuncOp"> {
  let summary = "A padded pool is a few unpadded pools on bands of its output";
  let description = [{
    A pool over a padded copy of a buffer reads a buffer that was filled with a
    constant and then had the real image copied into the middle of it. GoogLeNet
    pays 1.3 million element moves an inference for that, and thirteen of its
    eighteen copies come straight out of the accelerator, which cannot be asked
    to write into the middle of a padded buffer: Gemmini's output has one
    stride, not a row stride and a pixel stride.

    So take the padding away. Output index `o` reads the real image at
    `o*s - lo + t*d` for tap `t`, and since the window slides by a constant
    stride the taps that fall outside the image change only near the edges --
    each distinct clipping holds over a contiguous run of `o`, a band. Each band
    gets its own pool with a smaller window, reading the real image directly.

    A band that touches padding has its output filled with the pad value first,
    so it computes `max(p, the real taps)` -- which is what the padded pool
    computed.

    It used to leave a pool `--pack-int8-max-pool` could pack alone: bands write
    strided subviews, the packing needed an identity layout, and losing eight
    channels to an `i64` cost far more than the copy this saves -- `googlenet`
    341.74 -> 444.68 ms, +30.1%. The packing reads a **band** now, so both
    apply: GoogLeNet's twelve padded copies, half a million element moves an
    inference, go to none, and every pool stays packed.
  }];
}

def ApproximateExp: Pass<"approximate-exp", "::mlir::func::FuncOp"> {
  let summary = "A softmax's exponential does not need libm";
  let description = [{
    `math.exp` lowers to a call to libm's `expf`, which is correctly rounded and
    carries `errno`. A decoder-only transformer spends 9.9% of its time there
    -- `expf32` plus the `xflowf`, `with_errnof` and `__math_uflowf` error
    paths -- for 3.3% of its elements, about 57 cycles an exponential.

    A softmax's exponential is divided by the sum of its siblings and the
    quotient is quantized to an i8, so those last bits are thrown away twice.
    MLIR's own polynomial approximation is a range reduction and a degree-6
    polynomial, accurate to a few ULP, with no call and no error paths.

    Only `math.exp`; `--convert-math-to-libm` still takes everything else.

    **Measured and not in the pipeline**: `gpt_tiny` +5.0%, `regnet_y_400mf`
    +1.7%, `vit_tiny` +0.6%. The accuracy licence held -- outputs were byte for
    byte identical, the difference being below the i8 the softmax is quantized
    to -- but this board's `expf` is a table and a short polynomial whose call
    pipelines, against 24 operations of which nine are a dependent `fma`.
  }];
}

def DepthwiseAsBlockDiagonal: Pass<"depthwise-as-block-diagonal", "::mlir::func::FuncOp"> {
  let summary = "A depthwise convolution is a block-diagonal one, DIM channels at a time";
  let description = [{
    `tiled_conv_dw_auto` sets `pochs` to 1, so a depthwise layer costs one
    accelerator call per channel. EfficientNet has some eight thousand channels
    across its sixteen depthwise layers and spends 30% of its time in that
    runtime, 20.8% of the model on a single `LOOP_CONV_WS` instruction.

    The array is 16x16 and a depthwise uses one column. Filling the other
    fifteen with zeros costs nothing in the array and turns DIM calls into one;
    what it costs is weight bytes. Measured on the board, a 240-channel 3x3
    depthwise over 8x8 went 2.885 ms in 240 calls to 0.443 ms in 15, with all
    15,360 output bytes identical.

    `lanes` must be the bitstream's DIM.
  }];
  let options = [
    Option<"lanes", "lanes", "int64_t", /*default=*/"16",
           "Channels a call; the accelerator's DIM">,
  ];
}

def RelaxFloatMaxPool: Pass<"relax-float-max-pool", "::mlir::func::FuncOp"> {
  let summary = "A float max-pool whose answer becomes an integer does not need IEEE maximum";
  let description = [{
    `linalg.pooling_nhwc_max` on floats is an IEEE maximum: it propagates a NaN,
    and LLVM expands that into a pair of `feq.s`/`bnez`/`fmv.s` around the
    comparison -- six instructions where `fmax.s` is one, on every step of every
    window. In GoogLeNet, whose two transition pools are the last still running
    in f32, the NaN test alone is 3.2% of the model.

    `arith.maxnumf` hands back the operand that is not a NaN and is one
    `fmax.s`. The two differ only when an operand is a NaN: one answers NaN and
    the other answers the number. That is legal exactly where the NaN would
    have reached an `arith.fptosi`, which is poison for a NaN -- so the pass
    asks that every reader of the buffer the pool writes carries the value to a
    conversion through operations that **propagate** a NaN.

    Which is a narrower question than "does it become an integer". An
    `arith.cmpf` answers `false` for a NaN and something else for a number, and
    that answer is defined; a relu spelled `arith.maxnumf` erases the NaN
    outright. Either one turns the difference into an ordinary wrong number, so
    both are refused.

    Not a trade against the dependency chain: five instructions out and none
    back.
  }];
}

def UnrollReductionWindows: Pass<"unroll-reduction-windows", "::mlir::func::FuncOp"> {
  let summary = "Straighten out a reduction loop whose trip count is a window size";
  let description = [{
    Once `--promote-reduction-accumulator` has taken the store-to-load round trip
    out of a reduction, what is left in the hot loops is almost pure
    `addi`/`li`/`add`/`blt`/`lui` -- the nest scaffolding. A 3x3 max-pool is a
    six-deep nest whose innermost loop runs **three times**, so the bookkeeping
    costs more than the nine loads and eight compares it exists to schedule.

    Unrolling those levels away leaves one straight run, and because the
    accumulator is already an iteration argument the run is a single chain with
    no memory in it.

    Only a loop that carries a value, so this stays on reductions, where the trip
    counts are window sizes; a plain elementwise leaf runs the length of a row
    and belongs to `--unroll-elementwise-loops`, which measured two iterations to
    be the right factor there. The budget is on the *unrolled* body: 3x3
    straightens completely, 7x7 straightens the inner level and keeps the loop
    above it.
  }];
  let options = [
    Option<"maxTrips", "max-trips", "unsigned", /*default=*/"8",
           "Largest trip count to straighten out">,
    Option<"maxOps", "max-ops", "unsigned", /*default=*/"96",
           "Largest body, in operations, to leave behind">,
  ];
}

def UnrollElementwiseLoops: Pass<"unroll-elementwise-loops", "::mlir::func::FuncOp"> {
  let summary = "Put two iterations of a leaf loop in one body";
  let description = [{
    The innermost loop of an f32/i8 conversion is a dozen instructions of which
    the work is a load, a multiply, a convert and a store -- a chain this
    in-order core walks with nothing to put in the gaps. More iterations in one
    body give the scheduler more independent chains to interleave.

    **Four, not two.** Two was measured first and four was measured to be no
    better -- on a pipeline where the loops were much longer, before the
    accumulator came out of memory, the windows were straightened and the
    per-channel affine was folded. Re-measured on the shorter chains those left:
    `vit_tiny` -5.2%, `googlenet` -4.6%, `densenet121` -4.4%,
    `efficientnet_b0` -3.8%, twelve of thirteen faster and none slower than
    +0.1%.

    **Eight where eight divides, four where it does not.** Re-measured on the
    board at the set's own shapes, with the unrolled forms checked byte for byte
    against the rolled one first: eight is -3.7% to -3.8% against four on all
    three bodies the set spends its elementwise time in, and sixteen buys
    another 0.3% to 1.9% for twice the code again.

    The reason four shipped rather than eight was that the factor has to divide
    the trip count so there is no epilogue, and a bigger factor therefore
    reaches fewer loops. That is an argument for trying eight **and then four**,
    not for picking one: `factor` is now the ceiling and is halved until it
    divides. A trip count of 32 takes eight, 12 takes four, 6 takes two, 17
    takes none -- exactly as before.

    Leaves only: a loop with another loop inside it is the nest's scaffolding,
    and a loop with a call in it is an accelerator or a copy whose cost is not
    the loop.
  }];
  let options = [
    Option<"factor", "factor", "unsigned", /*default=*/"8",
           "Most iterations to put in one body; halved until it divides">,
  ];
}

def SelectToMinMax: Pass<"select-to-minmax", "::mlir::func::FuncOp"> {
  let summary = "Rewrite a compare and a select as arith.maxnumf / arith.minnumf";
  let description = [{
    A relu arrives from torch-mlir as `arith.cmpf ugt` and `arith.select`, and
    lowers to a compare, a data-dependent branch, a move and a jump. `fmax.s` is
    one instruction and one step on the dependency chain.

    They agree exactly when the operand the select would yield on a NaN is not
    one -- the false value for an ordered predicate, the true value for an
    unordered one -- which the pass proves rather than assumes. It also leaves
    alone a maximum against a negative zero and a minimum against a positive
    one, where `fmax.s` and the select disagree on the sign.

    Run **after** `--convert-linalg-to-gemmlir`, which reads the compare to
    recognise an activation it can fold into the accelerator.
  }];
  let options = [
    Option<"licenseByDestination", "license-by-destination", "bool",
           /*default=*/"true",
           "Also rewrite a select whose result reaches nothing but a conversion "
           "to an integer, where a NaN is poison either way. Measured twice: "
           "2.9% slower on the long loops of 2026-09-14, and DenseNet-121 "
           "-3.8% on the short ones of 2026-09-15.">,
  ];
}

def ReciprocalForDivision: Pass<"reciprocal-for-division", "::mlir::func::FuncOp"> {
  let summary = "Rewrite division by a constant as multiplication by its reciprocal";
  let description = [{
    On this in-order core `fdiv.s` is most of a quantization loop -- 22 cycles an
    element of a loop that is otherwise about thirteen instructions.

    Run **after** `--convert-linalg-to-gemmlir`: the requantization matchers read
    the `divf` to recover the scale.
  }];
}

def HoistInvariantReciprocal: Pass<"hoist-invariant-reciprocal", "::mlir::func::FuncOp"> {
  let summary = "Divide once per row, not once per element";
  let description = [{
    A division whose divisor is read through an indexing map that drops one of
    the nest's loops is the same division over and over. One reciprocal per
    distinct divisor, then a multiply inside.

    `fdiv.s` is about 22 cycles here and does not pipeline, so a softmax paying
    it once per element of its score matrix is most of the loop. On `sa` -- one
    head of self-attention, 64 tokens and 64 channels -- **20.81 to 17.84 ms**,
    relative L2 unchanged.

    Confined to a quotient that reaches a conversion to an integer, which is
    what makes the runtime reciprocal's edge cases unobservable; the pass
    comment argues both of them. Buffer semantics, and **before**
    `--plan-static-buffers`, which is what keeps the reciprocal's buffer off the
    allocator.
  }];
}

def FoldConstantElementwise: Pass<"fold-constant-elementwise", "::mlir::ModuleOp"> {
  let summary = "A loop whose every input is a constant is a constant";
  let description = [{
    An elementwise `linalg.generic` on buffers whose every `ins` operand is a
    constant `memref.global` writes the same numbers on every inference.
    Evaluate it once, here, and leave a constant global in its place.

    What it finds are the per-channel loops MID itself creates.
    `--hoist-invariant-reciprocal` lifts `rsqrt(var[c] + eps)` out of a batch
    norm that had no contraction to fold into, and `--combine-channel-affine`
    turns the four numbers that follow into two -- both loops read nothing but
    the layer's parameters. DenseNet-121 runs 31,616 steps of them per
    inference, and `fsqrt.s` with `fdiv.s` is **5.3% of that model** by
    program-counter sampling.

    Nothing in FRONT could have seen them: the constant folding there runs
    before the passes that write these loops exist.

    The evaluator is deliberately small -- `addf`, `subf`, `mulf`, `divf`,
    `negf`, `truncf`, `extf`, `sqrt` and `rsqrt` on `f32`/`f64`, and a float
    `arith.constant` -- and anything else refuses rather than guesses. Every
    one of those is correctly rounded in IEEE-754 on the host and on the board,
    and an `f32` value is rounded back through `float` after each step, so the
    constant is bit for bit what the loop would have computed. `math.rsqrt` is
    evaluated as `1.0 / sqrt(x)`, which is how `--math-expand-ops=ops=rsqrt`
    spells it in LOWER.

    The output buffers must be `memref.alloc`s this loop is the only writer of,
    with no aliasing name taken of them.
  }];
  let options = [
    Option<"maxElements", "max-elements", "int64_t", /*default=*/"8192",
           "Largest buffer to turn into constant data, in elements">
  ];
}

def SinkElementwiseIntoReaders: Pass<"sink-elementwise-into-readers", "::mlir::func::FuncOp"> {
  let summary = "A buffer read twice is a buffer written once too often";
  let description = [{
    An elementwise `linalg.generic` on buffers whose result is read by a handful
    of other loops and by nobody else: recompute it in each of them, and the
    buffer -- the write and every read of it -- goes away.

    A ViT's layer norm is the shape this was built for. It writes `x - mean`
    into a 17x192 f32 buffer, 13 KB against a 16 KB L1, and reads it twice: once
    for the variance and once to normalize. Upstream elementwise fusion refuses
    it because fusing wants `hasOneUse` and this has two. Measured as a kernel
    on the board, recomputing the subtraction in both readers is **-12.8%** on
    the layer norm, and byte for byte the same answer: the same arithmetic in
    the same order, only not stored in between.

    Recomputing is not always cheaper, and the bound is **traffic, not count**.
    DenseNet's dequantize feeds the batch norm of every later layer in its dense
    block -- up to 25 readers -- and doing it 25 times instead of once measured
    +0.9% on the board. Counting bytes,

    ```
      before = sum(inputs) + write(B) + N * read(B)
      after  = N * sum(inputs)
    ```

    so it pays exactly when `(N-1) * sum(inputs) < (N+1) * size(B)`, on top of a
    cap on the number of readers and on the size of the body -- the arithmetic
    the recomputation repeats is not in that inequality.

    Sound because nothing is reordered: the producer's inputs must be provably
    unwritten between it and each reader, the buffer must have no other writer
    and no alias taken of it, and the body must be memory-effect-free with no
    `linalg.index`, which reads the loop it sits in.

    Buffer semantics, and **before** `--plan-static-buffers`, so the buffer that
    disappears is never given a slot.
  }];
  let options = [
    Option<"maxConsumers", "max-consumers", "int64_t", /*default=*/"3",
           "Largest number of readers to recompute into">,
    Option<"maxOps", "max-ops", "int64_t", /*default=*/"4",
           "Largest producer body to recompute, in operations">
  ];
}

def CombineConstantScales: Pass<"combine-constant-scales", "::mlir::func::FuncOp"> {
  let summary = "Two constant coefficients on the dependency chain are one";
  let description = [{
    A quantization tail carries its constants one operation at a time:

    ```
      fcvt.s.w  fa3, a1              # the accumulator
      fmadd.s   fa3, fa3, fa5, fs1   # * input scale + bias
      fmul.s    fa3, fa3, fa4        # / output scale
      fcvt.w.s  s0, fa3, rne
    ```

    Both coefficients are known at compile time and both sit **on the dependency
    chain**, which is what an in-order single-issue core pays for. `(a*x + b)*k`
    is `a*k*x + b*k`: one operation, with the constants folded once by the
    compiler.

    Collapses a maximal single-use chain of `mulf`, `divf`, `addf`, `subf`,
    `negf` and `math.fma` by compile-time constants into one `math.fma`, or a
    single `arith.mulf` when the offset comes out zero.

    The reassociation is licensed the way the reciprocal is -- the value reaches
    nothing but a conversion to an integer, through monotone operations only, so
    a few ulps can move nothing but a rounding boundary. It is refused outright
    when the folded coefficients stop being finite or the scale collapses to
    zero, which is the one way the answer could change by more than that.
  }];
}

def FoldScalesIntoBroadcast: Pass<"fold-scales-into-broadcast", "::mlir::func::FuncOp"> {
  let summary = "Scale the operand there are fewer of";
  let description = [{
    EfficientNet's squeeze-and-excitation is

    ```
      (x * input_scale) * gate[c] / output_scale
    ```

    -- three floating-point multiplies an element, two of them by numbers the
    compiler knows. `gate` is **96 numbers** against 16x16x96 elements, so the
    two constants belong in it: one small loop, and the element loop drops from
    four operations to two. It is 26% of that model's elementwise work.

    `--combine-constant-scales` cannot reach these, because the running
    coefficient sits *between* the two constants and breaks the chain. The rule
    is the one `--combine-channel-affine` used: when a loop multiplies by
    something it reads through a map that covers fewer elements than the loop
    has steps, the constants around that multiply belong on its side.

    Only a **pure scaling** moves -- an offset does not commute with the
    multiply -- and the licence is the usual one: the value reaches nothing but
    a conversion to an integer, so a few ulps move only a rounding boundary.

    Extending it to carry a *constant* offset as well -- `(x*r + b)/s` is
    `x*(r/s) + b/s` -- was written, tested and reverted: it fires on a layer
    norm, but `--combine-constant-scales` already reaches the same end state
    whenever `beta` is zero, which is every layer norm in the set. Identical
    floating-point operation counts in the final IR, and one object changed for
    nothing.

    Buffer semantics, and before `--plan-static-buffers`, which is what keeps
    the scaled copy off the allocator.
  }];
}

def BatchNormInFixedPoint: Pass<"batch-norm-in-fixed-point", "::mlir::func::FuncOp"> {
  let summary = "A batch norm on a dequantized byte is integer arithmetic";
  let description = [{
    DenseNet's batch norm is 72% of the model and about 27 cycles an element for
    an eight-instruction body. Shortening it was measured five ways -- fewer
    loads, fewer input bytes, a contiguous output, a different loop order, the
    relu moved -- and every one bought 1-2%. What it is paying for is the
    **floating point**.

    `--fold-constant-elementwise` made the per-channel coefficients compile-time
    constants, and the value they multiply is a dequantized byte, so the whole
    tail is a function of one byte and one channel:

    ```
      lb / mul / add / srai / clamp / sb     instead of
      flw flw flw / fmadd / fmax / fcvt.w.s / clamp / sb
    ```

    Measured as a kernel on the board at dense block 3's shape: **-16.6%**.

    **And it is exact, not approximately exact.** Both coefficients and the
    dequantize scale are known at compile time and the input is one of 256
    bytes, so the pass evaluates *both* forms for every channel and every byte
    and rewrites only where they agree everywhere. The shift is the largest one
    that cannot overflow an i32; where no shift is precise enough, the site
    keeps its floating point.

    Runs after `--fold-constant-elementwise`, which is what makes the
    coefficients constants, and before `--plan-static-buffers`, so the
    dequantized buffer it leaves unread is never given a slot.
  }];
}

def PackInt8TableLookup: Pass<"pack-int8-table-lookup", "::mlir::func::FuncOp"> {
  let summary = "Eight indices out of one load";
  let description = [{
    An i8 -> i8 table sweep is **72% of EfficientNet's elementwise work** and 29%
    of a ViT's. `--table-for-i8-elementwise` leaves it as four instructions an
    element -- `lb`, `add`, `lbu`, `sb` -- and it measures about **11 cycles**,
    so something is waiting. The chain is two *dependent* loads: the byte, and
    then the table entry that byte indexes.

    One 8-byte load takes the first of them out of the chain. The bytes come out
    of the register with a shift and a mask, so the body grows from four
    instructions to about five and a quarter -- and measured on the board at
    EfficientNet's shape, 16x16x576: **-44.5%**. Packing the eight *stores* back
    into one word as well is worse (-37.7%), so the outputs stay bytes.

    The index the table pass computes is `sext(q) + 128`, which for the unsigned
    byte `u` sitting in the word is `u ^ 0x80` -- so a single `xor` of the whole
    word does all eight indices at once.

    Needs both buffers contiguous and identity-laid-out, which is why it runs
    late, next to `--pack-int8-max-pool`.
  }];
}

def ClampAsRangeCheck: Pass<"clamp-as-range-check", "::mlir::func::FuncOp"> {
  let summary = "Two clamps are one unsigned compare";
  let description = [{
    Every quantization tail ends `maxsi(v, lo)` then `minsi(v, hi)`. This board
    is plain rv64gc -- no `Zbb`, so no `max`/`min` and no conditional move -- and
    the pair costs five instructions:

    ```
      sgtz a5, a4 / neg a5, a5 / and a5, a5, a4 / li a4, 127 / blt a5, a4, .+
    ```

    But the pair is a **range check**, and one *unsigned* compare decides it:
    anything outside `[lo, hi]` has `(unsigned)(v - lo) > hi - lo`, negatives
    included. The fixup goes out of line, where it is rarely taken, and the
    common path is one `bltu`.

    Measured as a kernel at DenseNet's batch-norm shape, where 774 of 4096
    values actually clamp: **-5.1%**. The clamp is 11.3% of `densenet121` by
    program-counter sampling, 10.2% of `vit_tiny` and 7.6% of
    `efficientnet_b0`.

    Emits `scf.if` rather than two `arith.select`s, because a select is
    branchless and branchless is exactly what costs five instructions here.
  }];
}

def UnrollAcceleratorLoops: Pass<"unroll-accelerator-loops", "::mlir::func::FuncOp"> {
  let summary = "A loop the flush analysis cannot see inside keeps every flush";
  let description = [{
    `--place-cache-flushes` models a region it cannot otherwise account for as
    one host step that reads and writes everything inside, and **never gives the
    accelerator calls in that region an attribute** -- so they keep both their
    flushes, and a flush is a walk of the whole L1.

    A transformer is the only shape in the set with such loops: a batch matmul
    lowers to one slice a trip, and `vit_tiny` has **24 of them**, each three
    iterations of nothing but a `gemmlir.matmul_i8`. `gemmlir_evict` is 6.0% of
    that model by program-counter sampling.

    Nothing in those loops is a host step at all. Unrolling them fully puts the
    calls in the straight line the analysis already handles, where a flush
    survives only where the host really did write in between.

    Sound because the flush is not the fence: `gemmlir.no_flush_*` controls only
    the cache walk, and the `fence` that waits for the accelerator to go idle is
    emitted on every call regardless.

    Runs in FRONT, immediately before `--place-cache-flushes`.
  }];
  let options = [
    Option<"maxTrips", "max-trips", "int64_t", /*default=*/"8",
           "Largest trip count to unroll">
  ];
}

def DropUnreadPadding: Pass<"drop-unread-padding", "::mlir::func::FuncOp"> {
  let summary = "Drop a padding a pooling operation never reads";
  let description = [{
    PyTorch's `ceil_mode` asks for an output one step wider than the input
    supports, and torch-mlir pays for it with a `tensor.pad` of `stride - 1` on
    the high side. Where the window divides the input evenly the ceiling and the
    floor agree and not one padded element is reached -- what is left is a buffer
    the host zero-fills and copies into for nothing, standing between the
    convolution and the pool, which is exactly what stops the pool riding out on
    the convolution's own `mvout`.

    The test is arithmetic and exact: the highest index the pool reads on a
    spatial axis is `(out - 1) * stride + (window - 1) * dilation`, and if that
    is inside the source on every axis -- with nothing padded on the low side,
    which would shift the indices -- the pad contributes nothing and the pool
    reads the source instead.

    SqueezeNet has three of them: 31x31 padded to 33x33, 15x15 to 17x17 and 9x9
    to 11x11.
  }];
  let dependentDialects = ["::mlir::func::FuncDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::tensor::TensorDialect"];
}

def RequantizeBeforePooling: Pass<"requantize-before-pooling", "::mlir::func::FuncOp"> {
  let summary = "Move a monotonic elementwise operation ahead of a max-pool";
  let description = [{
    `max` commutes with any non-decreasing function: `max(f(a), f(b))` is
    `f(max(a, b))`. A quantized network pools in f32 and requantizes afterwards,
    which leaves the convolution's tail -- dequantize, bias, activation -- with
    nothing to fold into and keeps the whole layer in software. Moving the
    requantization ahead of the pool gives the convolution an i8 result to fold
    into, and the pool then runs on i8 as well.

    Only a body built out of operations that are provably non-decreasing is
    moved: scaling by a positive constant, adding a constant, rounding,
    widening, clamping, and a truncation that a clamp to the destination's range
    has already made safe.

    **A pool with a second reader is moved too.** GoogLeNet's inception pool
    feeds the requantization *and*, through a pad, the next module's pool; and
    once that one has been moved, the second reader is the very same
    requantization. `hasOneUse` refused both of the two pools that mattered --
    exactly the ones whose input is a concatenation of four dequantized
    branches, which is what kept 99,072 elements of f32 dequantize, 77,184 of
    requantize and a 276 KB copy in the model.

    Such a reader is handed the **inverse** of the quantization that moved: one
    multiply by the scale it divided by. That is exact where it matters --
    `quantize(dequantize(q))` is `q` for a byte -- so a reader that quantizes
    again, or that is another max-pool feeding one, sees no difference. Only
    the divide-by-a-constant body has an inverse this pass can write, which is
    why the multi-reader case is confined to it.

    On `googlenet` it takes all thirteen pools to i8 rather than eleven, and
    with the pool no longer in f32 the convolutions above it have an i8 result
    to fold into: **eight im2col matmuls become native convolutions** and the
    model's elementwise work falls from 446,824 element-visits to 28,648.
  }];
}

def RoundQuantizedCasts: Pass<"round-quantized-casts", "::mlir::func::FuncOp"> {
  let summary = "Round to nearest before the integer conversion of a quantize";
  let description = [{
    `--lower-quant-ops` turns `quant.qcast` into `fptosi(divf(x, scale))`, and
    `arith.fptosi` truncates toward zero. Quantization is round-to-nearest, and
    the difference dominates everything else: on a PyTorch MLP run on the board
    it was the whole gap between a relative L2 error of 0.036 and 0.012.

    Only a float divide feeding an integer conversion is rewritten, so a
    truncation the input asked for on its own is left alone. Run it after
    `--lower-quant-ops`.
  }];
}

// Convolution reaches the accelerator as a matmul, which is how the reference
// workloads run it too.
def ConvToImg2Col: Pass<"conv-to-img2col", "::mlir::func::FuncOp"> {
  let summary = "Rewrite convolutions as im2col packing plus a matmul";
  let description = [{
    Wraps MLIR's `populateConvertConv2DToImg2ColPatterns`, which handles the
    NHWC and NCHW named convolutions on tensors.

    This sidesteps the layout problem rather than solving it. A frontend's
    convolution does not arrive in the layout `tiled_conv_auto` wants -- PyTorch
    lowers to `conv_2d_nchw_fchw`, TOSA to `conv_2d_nhwc_fhwc`, and the runtime
    reads (KH, KW, C, F) -- but after im2col there is no layout question left,
    only a matmul, which is the best-supported path here. It is also what
    gemmini-rocc-tests does: its ResNet and MobileNet inputs are stored already
    packed, and 87 of 110 of their accelerator calls are matmuls.

    Run it before `--convert-linalg-to-gemmlir`, on tensors.

    Off by default in the quantized pipeline: a convolution that ends in a
    requantization folds into `conv2d_i8`, which takes the bias, the activation
    and the pooling with it and beats the pack-plus-matmul on this board. Use
    `unfoldable-only` to convert just the ones that cannot.
  }];
  let options = [
    Option<"unfoldableOnly", "unfoldable-only", "bool", /*default=*/"false",
           "Only convolutions that could not fold into `conv2d_i8` anyway: "
           "int8 in, int32 out, and a single consumer that widens to f32.">,
  ];
}

// A batch-of-one contraction is a matmul with extra axes.
def RaiseSpatialSumToPool: Pass<"raise-spatial-sum-to-pool", "::mlir::func::FuncOp"> {
  let summary = "A sum over both spatial axes is a global pool";
  let description = [{
    `nn.AdaptiveAvgPool2d(1)` arrives as `linalg.pooling_n*_sum` and everything
    downstream knows it. `x.mean(dim=(2, 3))` -- how most PyTorch code writes the
    same thing -- arrives as a bare `linalg.generic` with two reduction
    iterators, and nothing matches it: the convolution above keeps its scalar
    loop and the layout rewrite leaves a transpose over the whole activation.

    Run **before** `--conv-nchw-to-nhwc`, so the pool is carried along with the
    rest and the transpose never appears.
  }];
}

def RaiseContractionToMatmul: Pass<"raise-contraction-to-matmul", "::mlir::func::FuncOp"> {
  let summary = "Rewrite a unit-batch contraction generic as linalg.matmul";
  let description = [{
    `--conv-to-img2col` leaves a `linalg.generic` contraction rather than a named
    operation when the batch is 1, and everything downstream -- the
    quantization, the gemmlir conversion -- matches named operations. This
    raises it: the unit batch axes are collapsed away and the result is a
    `linalg.matmul`, carrying `indexing_maps` when an operand is transposed.

    Only contractions with one m, one n and one k dimension, whose batch axes
    are all of extent 1 and whose operands are plain 2-D views of those
    dimensions, are raised. `--linalg-fold-unit-extent-dims` covers more ground
    but miscompiles this IR -- see docs/pipeline.md.
  }];
}

// Elementwise fusion that leaves the matmuls alone.
def FuseElementwiseAroundMatmul: Pass<"fuse-elementwise-around-matmul", "::mlir::func::FuncOp"> {
  let summary = "Fuse elementwise operations without fusing into a contraction";
  let description = [{
    The quantization pipeline leaves a long chain of elementwise passes over
    each activation -- dequantize, bias, activation, requantize -- and on a
    small CNN almost all of the run time is those loops rather than the
    accelerator. Fusing them is worth a lot; fusing a quantization *into* the
    matmul that consumes it is not, because the matmul stops looking like a
    matmul and is never offloaded. Measured with the stock
    `--linalg-fuse-elementwise-ops`: four offloaded matmuls became one, and 35 ms
    became 232 ms.

    So this runs the same patterns with a control that refuses any fusion whose
    consumer is a contraction. Run it on tensors, before bufferization.
  }];
}

// Quantize before the packing, not after it.
def HoistElementwiseBeforeGather: Pass<"hoist-elementwise-before-gather", "::mlir::func::FuncOp"> {
  let summary = "Move an elementwise operation to the other side of a pure gather";
  let description = [{
    `elementwise(gather(x))` and `gather(elementwise(x))` compute the same
    values -- a gather only moves elements around -- but the second does the
    work on the *unexpanded* operand.

    That matters after `--conv-to-img2col`: the packing expands its input by the
    kernel footprint, so quantizing afterwards converts nine times as many
    elements for a 3x3 kernel, and the packing then copies f32 instead of i8.
    The ordinary elementwise fusion cannot do this -- im2col's index map has
    floordiv and mod in it, so it is not a permutation.

    Applies when the gather's body only yields its input, the elementwise
    operation has identity maps and does not read its output, and nothing else
    uses the gathered values.
  }];
}

// Quantize once at a branch point and share the result.
def ShareBranchQuantization: Pass<"share-branch-quantization", "::mlir::func::FuncOp"> {
  let summary = "Quantize a branching activation once instead of per consumer";
  let description = [{
    A residual block's activation has two consumers: the next convolution, which
    wants it quantized, and the shortcut, which wants it in f32. So
    `--force-quantized-matmul` leaves the producing convolution's tail in f32 and
    puts a separate `quant.qcast` in front of the convolution. That tail then
    never ends in a requantization, `--convert-linalg-to-gemmlir` has nothing to
    fold the convolution into, and the convolution stays a scalar loop.

    This quantizes the branch point itself and hands every other consumer the
    dequantization of that one value -- which is what a quantized residual
    network does anyway, and is the precondition for reaching `resadd_i8`. The
    producing convolution is then left with a single consumer that ends in a
    requantization, so it folds.

    The `quant.qcast` is usually not on the branch value itself: a frontend pads
    and transposes on the way into the convolution. Those are rebuilt on i8,
    which is where they wanted to be anyway -- the padding value has to
    quantize to itself for that, which for a symmetric quantization means zero.

    Only fires at a real branch (the value has more than one use) whose value
    comes off a quantized contraction, and never on a value that is already a
    dequantization -- that last condition is what terminates the rewrite, since
    the result of one is exactly the input shape of another.
  }];
}

// Take the residual out of the convolution's tail.
def SplitResidualAdd: Pass<"split-residual-add", "::mlir::func::FuncOp"> {
  let summary = "Separate a residual add from the requantization it is fused with";
  let description = [{
    In a residual block the second convolution's tail *is* the add: one
    `linalg.generic` reads the shortcut, the accumulator and the bias and writes
    the block's i8 output. `--convert-linalg-to-gemmlir` reads a requantization
    of one accumulator, so it cannot fold that, and the convolution stays a
    scalar loop -- which on a small ResNet block is most of the run time.

    This splits it in two: the convolution's own requantization to i8, which
    folds into the call, and the scaled add of two i8 tensors, which is what
    `gemmlir.resadd_i8` is. The intermediate is quantized at the block output's
    scale, so the add's own scaling is exact and only the convolution's result
    is rounded -- but rounded it is, and clipped to i8 before the add, which the
    reference computes in i32. That is a real approximation and the reason this
    is a separate pass: it is what a quantized residual network does, and it is
    measured rather than assumed.
  }];
}

// An average pool is a contraction against a constant.
def AveragePoolToContraction: Pass<"average-pool-to-contraction", "::mlir::func::FuncOp"> {
  let summary = "Rewrite a sum pool as the contraction it is";
  let description = [{
    A frontend lowers `adaptive_avg_pool2d(x, 1)` -- the head of every ResNet and
    MobileNet -- to a `linalg.pooling_nhwc_sum` over the whole image followed by
    a divide. Gemmini has a `tiled_global_average_auto`, but it produces the
    average at the *input's* scale, which for a mean over hundreds of pixels
    throws away most of the output's range.

    Summing a whole image per channel is a contraction: with the image seen as
    `(H*W, C)`, it is `ones(1, H*W) x image`. That form goes down the path that
    already exists -- quantized against a folded constant, accumulated in i32,
    and requantized at the output's own calibrated scale by the same fusion that
    serves every other layer -- so the mean is exact and the divide by the pixel
    count disappears into the accelerator call's scale.

    A pool with a *window* rather than the whole image is a depthwise
    convolution whose filter is all ones, which is what `tiled_conv_dw_auto`
    runs. It matters for the same reason: a sum does not commute with a
    quantization, so `--requantize-before-pooling` cannot move the
    requantization across it the way it can across a max pool, and the
    convolution above stays a scalar loop.

    Runs before calibration, since the operations it produces are ones the
    calibrator annotates. A batch of images keeps its batch dimension and
    becomes a `linalg.batch_matmul` -- collapsing it away would sum across
    images.
  }];
}

// A per-channel affine after a contraction belongs in the weights.
def FoldBatchNorm: Pass<"fold-batch-norm", "::mlir::func::FuncOp"> {
  let summary = "Fold a per-channel affine into the weights of what produced it";
  let description = [{
    A batch norm in evaluation mode is a per-output-channel affine,
    `out = in * a + b` with `a = gamma / sqrt(var + eps)` and
    `b = beta - mean * a`, and torch-mlir emits it as one `linalg.generic` over
    four constant vectors. Nothing downstream can take it: `tiled_conv_auto`
    scales its accumulator by a single number, not one per channel, so
    `--convert-linalg-to-gemmlir` sees a requantization it cannot read and the
    convolution stays a scalar loop. Every real CNN has one after every
    convolution.

    All of it is constant, so it goes into the weights: scaling output channel
    `f` of the filter by `a[f]` scales that channel's whole result, and the
    additive part joins whatever the convolution was already accumulating onto.
    What is left is an ordinary convolution.

    The body is not pattern-matched against the batch-norm formula. Each value
    in it is classified as constant or as `a * in + b`, and anything that is not
    affine in the activation -- or is built from an operation this does not know
    how to evaluate -- is refused. That covers a batch norm written any of the
    ways a frontend spells it, and also the plain per-channel scale a fused
    quantization leaves.
  }];
}

// A channel shuffle is a permutation of the weights.
def FoldChannelShuffle: Pass<"fold-channel-shuffle", "::mlir::func::FuncOp"> {
  let summary = "Fold a channel permutation into the weights that read it";
  let description = [{
    ShuffleNet's unit ends by interleaving the channels of the two halves it
    joined, which a frontend writes as `expand_shape` splitting the channel axis
    into (groups, rest), a transpose of those two, and `collapse_shape` putting
    them back. It is a fixed permutation of one axis, and on the accelerator it
    would be a whole pass over the activation to move data that is only going to
    be multiplied by a constant.

    A convolution reading a permuted input computes the same thing as the same
    convolution reading the original with its filter's input channels permuted
    the other way, and the filter is a constant. So the shuffle is done once, at
    compile time, in the weights -- and nothing moves at run time.

    Runs before `--conv-nchw-to-nhwc`, on the shape torch-mlir emits.
  }];
}

// A 1x1 convolution whose result stays in f32 is a matmul.
def PointwiseConvToMatmul: Pass<"pointwise-conv-to-matmul", "::mlir::func::FuncOp"> {
  let summary = "Rewrite a 1x1 convolution that cannot requantize as a matmul";
  let description = [{
    `tiled_conv_auto` writes `elem_t`: the accelerator's convolution always
    requantizes its accumulator to i8 on the way out, and `gemmlir.conv2d_i8` is
    the only shape there is. A convolution whose result the model *returns* has
    no requantization to fold -- the value leaves in f32 -- so it has nowhere to
    go and stays a scalar loop. Every detector and every model with an auxiliary
    head ends in exactly that.

    A 1x1 convolution over NHWC is a matmul: the pixels are the rows and the
    channels are the contraction. `tiled_matmul_auto` does have a form that
    writes the i32 accumulator, so as a matmul it offloads and the
    dequantization after it is the same scalar tail every model's last layer
    has.

    Only where it would not fold as a convolution. A 1x1 convolution followed by
    a requantization is left alone: `conv2d_i8` takes the bias, the activation
    and a fused pooling with it, and on this board the convolution call is the
    faster of the two.
  }];
}

// Drop the cache flushes that cannot matter.
def OrderLoopsForLocality: Pass<"order-loops-for-locality", "::mlir::func::FuncOp"> {
  let summary = "Put the axis with the smallest byte stride innermost";
  let description = [{
    An elementwise loop nest may run its axes in any order, and bufferization
    leaves them in the destination's. Where the operation also relayouts, that
    makes the source's slowest axis the innermost loop -- every load its own
    cache line, used for a few of its bytes.

    Sorting the axes by total byte stride, biggest outermost, is the ordinary
    rule. Run on memrefs, before `--convert-linalg-to-loops`.
  }];
}

def PlaceCacheFlushes: Pass<"place-cache-flushes", "::mlir::func::FuncOp"> {
  let summary = "Mark the accelerator calls whose cache flushes can be skipped";
  let description = [{
    Gemmini reads through the L2 and does not probe the host's L1, in either
    direction: the accelerator reads past stores the host still holds dirty, and
    the host reads lines the accelerator has since overwritten. `gemmlir_rt.c`
    displaces the L1 on both sides of every call, which costs about 0.12 ms a
    call -- 1.8 ms of ResNet-20's 11.3.

    Most of those flushes cannot matter. Between two convolutions of a fused
    network the host touches nothing at all, so there are no dirty lines to
    push out and no stale ones to drop. This pass works out which flushes are
    load-bearing and marks the rest `gemmlir.no_flush_before` /
    `gemmlir.no_flush_after`; the lowering emits a call only where the attribute
    is absent, so anything that does not run this pass keeps every flush.

    It decides by checking, not by pattern-matching: it starts from all flushes
    present -- which is the behaviour it has to preserve -- and tries removing
    them one at a time, keeping a removal only when simulating the function
    still shows no accelerator reading a host-dirty buffer and no host reading
    a buffer the accelerator has overwritten. The simulation runs the block
    repeatedly until the state repeats, because the function is called in a
    loop and the host's lines outlive one call; that is the whole reason the
    bug it guards against stayed hidden for so long.
  }];
  let options = [
    Option<"dropBefore", "drop-before", "bool", "true",
           "Allow removing the flush that runs before a call">,
    Option<"dropAfter", "drop-after", "bool", "true",
           "Allow removing the flush that runs after a call">,
  ];
}

// A grouped convolution is G convolutions that do not talk to each other.
def SplitGroupedConv: Pass<"split-grouped-conv", "::mlir::func::FuncOp"> {
  let summary = "Rewrite a grouped convolution as one convolution per group";
  let description = [{
    `linalg.conv_2d_ngchw_gfchw` is what torch-mlir emits for
    `nn.Conv2d(..., groups=G)` -- the grouped convolution of ResNeXt, RegNet
    and every "cardinality" block. Gemmini has no notion of groups, and it does
    not need one: group `g` reads only input channels `[g*C/G, (g+1)*C/G)` and
    writes only output channels `[g*F/G, (g+1)*F/G)`, so the whole thing is G
    convolutions that never see each other's data.

    This pass writes it that way: G `linalg.conv_2d_nchw_fchw` over
    `tensor.extract_slice`s of the input and the filter, each one written back
    with a `tensor.insert_slice`. From there the ordinary path applies -- NHWC,
    quantization, and the channel slices folding into the `in_stride` and
    `out_stride` of `tiled_conv_stride_auto`, which is the same mechanism a
    concatenation already uses.

    Depthwise convolution is the `G == C` end of this and torch-mlir emits its
    own operation for it, which has its own Gemmini call; this is the middle,
    where the groups are wider than one channel.
  }];
}

// Keep a padded convolution's input out of the padding's own buffer.
def FillToMemset: Pass<"fill-to-memset", "::mlir::func::FuncOp"> {
  let summary = "Rewrite a fill of a uniform byte as gemmlir.memset";
  let description = [{
    A convolution's padding bufferizes into a fill of the whole padded buffer
    plus a copy of the real input into the middle of it, and the fill is a
    scalar store per element. One repeated byte is a `memset` instead.

    Run on memrefs, **after** `--convert-linalg-to-gemmlir`: a zero fill is how
    an accumulator is proved to start at zero, and the conversion reads them.

    **A run per row is still a memset per run.** `--fill-only-the-border` cuts
    the fill into slabs, and the slab that is the last *columns* of every row is
    strided: `[1, 48, 2, 64]` over strides `[160000, 3200, 64, 1]` is 48 runs of
    128 bytes. Refusing it left the whole slab a store per byte. It becomes a
    loop over the rows with one `memset` inside, which is exact -- the same
    bytes in the same places -- and only when the run is at least a cache line,
    because below that the call costs more than the stores it saves.

    **Twice in the pipeline.** `accumulatedIntoLater` refuses a fill that a
    later `linalg` operation reads back as its init, which is what a max pool's
    neutral value is. `--pack-int8-max-pool` then turns that pool into an
    `scf.for` over words and the init stops being a `linalg` operand, so a
    second run after the packer converts it. On GoogLeNet the two runs together
    take 34 fills of 364,100 elements to **none**: every padding and every pool
    neutral is a `memset`, where PC sampling had put 12.4% of the model in
    loops that do nothing but store.
  }];
  let options = [
    Option<"belowReduction", "below-reduction", "bool", /*default=*/"false",
           "Also convert a fill whose buffer a later linalg reduction reads. "
           "The refusal was measured before --promote-reduction-accumulator "
           "existed, when the reduction still carried its accumulator through "
           "memory; with it on, GoogLeNet is -1.7%. The pipeline sets it.">,
  ];
}

def GatherToMemrefCopy: Pass<"gather-to-memref-copy", "::mlir::func::FuncOp"> {
  let summary = "Rewrite a copying linalg.generic as memref.copy on a strided view";
  let description = [{
    `out[i] = in[map(i)]` with a `map` that is linear in the iteration indices
    is a copy between two strided views of the same shape. Written that way the
    runtime copies the longest run that is packed on both sides with one
    `memcpy`, instead of the load-and-store per element
    `--convert-linalg-to-loops` emits.

    im2col is the one that matters: its read is
    `(n, oh, ow, kh, kw, c) -> (n, oh*sh + kh*dh, ow*sw + kw*dw, c)`, so with
    unit stride and dilation the packed run is `KW*C` and a pack of 18432
    elements becomes 768 `memcpy`s of 24 bytes.

    Run on memrefs, after `--convert-linalg-to-gemmlir` and before
    `--convert-linalg-to-loops`.
  }];
}

def MaterializePadSources: Pass<"materialize-pad-sources", "::mlir::func::FuncOp"> {
  let summary = "Give a padded convolution's producer a buffer of its own";
  let description = [{
    `tensor.pad` normally bufferizes into an allocation, a zero fill and a copy
    of the real input into the middle, and `--convert-linalg-to-gemmlir` turns
    exactly that shape into the runtime's own `padding`, so none of it survives.

    Once the padding is on i8 -- which is what `--hoist-elementwise-before-gather`
    makes of it, and much cheaper than padding f32 -- bufferization sees that it
    can write the producer straight into the middle of the padded buffer and
    skips the copy. That leaves a 16x16 window of an 18x18 buffer, and the
    runtime cannot address it: it takes one stride between pixels and derives
    the row from it. Both ends are then stranded, the producer because its
    destination is not addressable and the consumer because the shape the
    padding folds out of is not there.

    So the producer is given a buffer of its own, with
    `bufferization.alloc_tensor`. The copy that comes back is removed again by
    the padding fold, along with the padded buffer and its fill, before anything
    runs.
  }];
}

// A batch of one is not a batch.
def UnbatchSingleMatmul: Pass<"unbatch-single-matmul", "::mlir::func::FuncOp"> {
  let summary = "Rewrite a batch_matmul over one batch as a plain matmul";
  let description = [{
    `--convert-linalg-to-gemmlir` gives a `linalg.batch_matmul` a loop over the
    batch and a `memref.subview` per slice. Everything that fuses into a matmul
    -- the requantization, the bias, the activation -- looks at what follows the
    matmul in its own block, and what follows this one is the end of the loop
    body. So a batched matmul keeps every one of its tails as a separate scalar
    pass over the tensor.

    A batch of one has nothing to loop over. Collapsing it to a plain
    `linalg.matmul` puts it back on the path everything else takes, and the
    tails fuse as they do for a convolution. A transformer is where this shows:
    torch-mlir gives all five of an attention head's contractions as
    `linalg.batch_matmul`, and at batch 1 that was five matmuls whose
    quantization and dequantization stayed scalar -- 7400 of the 8480 scalar
    elements in one head.

    Batches larger than one keep the loop: the tails would have to be sliced to
    follow them in, which is a different transformation.
  }];
}

#endif // GEMMLIR_PASS

// A matmul whose accumulator three different quantizations read.
def SplitMatmulPerRequantize: Pass<"split-matmul-per-requantize", "::mlir::func::FuncOp"> {
  let summary = "Give each reader of a matmul's accumulator its own call";
  let description = [{
    The accelerator requantizes on the way out, so a matmul whose i32
    accumulator is read by one requantization becomes a `matmul_i8_scale` and
    the loop disappears. A transformer's attention projection is the shape that
    never gets there: one matmul writes Q, K and V side by side in one
    accumulator and three readers take a third of the columns each, at three
    different scales. One buffer, three scales, so no single call can serve it
    and all three requantizations stay on the host -- 43% of what ViT-Tiny still
    does outside the accelerator.

    A column slice of the product is the product of the same rows against a
    column slice of the weights, and `tiled_matmul_auto` already reads its
    operands through a row stride, so the slice needs no copy: it is a
    `memref.subview` of the weight matrix. Three calls, each with its own scale,
    each writing i8.

    What is left for the host is the relayout the reader also did -- the heads
    come out of the columns -- and that is now a byte copy of the same shape
    instead of a convert, a multiply, a round and a clamp per element.

    Only where it pays: two readers or more, none of them overlapping, and
    together no wider than the accumulator, so the split never multiplies the
    arithmetic. The scale each call gets is the reader's own scale folded with
    the dequantization above it, in f32 and in the same order
    `--combine-constant-scales` would have folded them, because that is the
    number the host would have multiplied by.
  }];
}

// A monotone map commutes with a maximum, so put it on the smaller side.
def SinkMonotoneBelowMaxPool: Pass<"sink-monotone-below-max-pool", "::mlir::func::FuncOp"> {
  let summary = "Move a monotone elementwise map from above a max pool to below it";
  let description = [{
    DenseNet's stem dequantizes its first convolution -- `sitofp`, a scale, a
    per-channel bias and a relu -- into a 32x32x64 f32 buffer, pads that into a
    34x34x64 one, and takes a 3x3 max pool of it. The dequantization runs on
    **four times as many elements as anything downstream ever reads**, through
    two buffers that exist only to be walked once.

    `max` commutes with a monotone non-decreasing map, so the pool can read the
    convolution's i32 accumulator directly and the dequantization can run once
    on the pooled result -- the same f32 arithmetic on the same number, because
    a strictly increasing map does not change which element wins.

    Measured as the stem's two loops at DenseNet's own shape: **-33.6%**, with
    every one of the 16,384 outputs identical.

    It costs the pool its `fmax.s`: a signed integer maximum with no Zbb is a
    data-dependent branch. That is what this was rejected for on paper, and the
    paper was wrong -- a branch on this core is far cheaper than a quarter of
    the elementwise work, which the clamp measurement in
    [[gemmlir-two-clamps-are-a-range-check]] says directly.

    The padding is the other half of the argument. The f32 pad holds the relu's
    own floor, and in the accumulator's domain the smallest representable
    integer comes back out of the map at exactly that floor -- so a padded
    position reads the same either way, and a window that is all padding gives
    the floor in both.

    Run after `--sink-elementwise-into-readers`, which is what puts the
    dequantization in one loop, and before `--plan-static-buffers`, so the two
    buffers it creates get arena slots, and before `--pool-without-padding`,
    which is happy to band an i32 pool.
  }];
}

// The mirror: many calls into one accumulator, one requantization over it.
def FoldRequantizeIntoSliceMatmuls: Pass<"fold-requantize-into-slice-matmuls", "::mlir::func::FuncOp"> {
  let summary = "Give each matmul writing a slice of an accumulator its own requantization";
  let description = [{
    `--split-matmul-per-requantize` handles one call and several readers. This
    is the other way round, and a transformer has both: the attention *output*
    is one matmul per head writing its own slice of a shared i32 accumulator,
    and a single requantization that reads the whole of it and puts the heads
    back where the model wants them.

    `FoldRequantizeIntoBatchMatmul` in the conversion already sinks a
    requantization into a per-slice matmul -- but only when the destination is
    laid out exactly like the accumulator, so a reader that also moves the head
    axis stops it. That reader is 14% of what ViT-Tiny still runs on the host,
    and it writes its result with a stride, which is the expensive way round.

    A slice of the accumulator has one destination block in the output, and the
    runtime addresses a block through a row stride -- so each call can write its
    own i8 directly, with the scale in the mvout, and the reader disappears
    entirely rather than turning into a copy. The permutation is only allowed to
    move whole axes: the block the call writes has to come out rank-2 with the
    same extents and unit-stride columns, which is what makes it addressable at
    all.

    **Not in the pipeline: the hardware writes that block wrong.** The rewrite
    itself is exact -- linked against the CPU runtime the whole model is byte
    for byte what it was -- but on the board a `matmul_i8_scale` whose output is
    a 64-wide block of a 192-wide buffer disagrees with that same runtime on
    every input. It is the matmul's version of
    [[gemmini-one-stride-per-pixel]]: a window narrower than its buffer is
    written to the wrong rows, and it is silent. Use this only where the
    destination is the whole width.
  }];
}

// A value that does not vary with the innermost loops belongs outside them.
def HoistBroadcastInvariants: Pass<"hoist-broadcast-invariants", "::mlir::func::FuncOp"> {
  let summary = "Compute a per-row value once per row, not once per element";
  let description = [{
    A layer norm's inner loop ends up as

    ```
      %s = arith.mulf %rstd, %scale     // %rstd is one number per token
      %y = arith.mulf %x_minus_mean, %s
    ```

    and `%rstd` is read through a map that names only the row. So `%s` is the
    same number for all 192 channels and the loop computes it 192 times. On
    `vit_tiny` that is about 78,000 floating-point multiplies an inference,
    against `fmul.s` being 10.2% of the whole model.

    Nothing else in the pipeline takes it: `--fold-scales-into-broadcast` moves
    *constants* onto the narrow operand and here both operands are already
    settled, `--combine-constant-scales` needs a chain of constants, and there
    is no loop-invariant code motion anywhere -- `llc` runs no IR pipeline at
    all ([[gemmlir-llc-is-not-minus-o2]]).

    So: find the operations in a body whose operands are all block arguments of
    operands read through maps that miss at least one of the loop's dimensions,
    or constants, and evaluate them in their own small loop over exactly those
    dimensions. The result is one more buffer of the narrow shape and one fewer
    operation an element.

    Only where it pays: the narrow shape has to be smaller than the loop, and
    the operations moved have to be worth more than the buffer they now go
    through -- which is why a single move of a single multiply still wins by
    the ratio, 192 evaluations becoming one.
  }];
}

// A two-dimensional maximum is two one-dimensional ones.
def SeparateMaxPool: Pass<"separate-max-pool", "::mlir::func::FuncOp"> {
  let summary = "Split a 2-D max pool into a horizontal pass and a vertical one";
  let description = [{
    **Built, tested, tried twice, and not in the pipeline.** The rewrite is
    exact -- `max` is associative and commutative, and both attempts were
    byte-for-byte against the CPU reference and against the build before -- and
    it is slower both times.

    A `kh x kw` window becomes `kh` rows of `kw` columns: eight comparisons an
    output become four for a 3x3, and the horizontal pass's result is shared
    between vertically adjacent outputs. A kernel at GoogLeNet's own shapes read
    **-64% to -68%**.

    **First attempt**, everything it matched: the set **+6.1%**, `googlenet`
    +28.4%, `squeezenet1_1` +14.1%, `densenet121` +8.8%. Two causes were
    identified and both were fixed:

    * at **stride 2** the row pass computes every row of the padded input where
      the direct form visits only `outH * kh` of them, and `outH * kh` is
      already the smaller number -- a loss on the arithmetic alone. Now refused.
    * the row buffer took a `linalg.fill` that was a **store per byte**. It is a
      `gemmlir.memset` now, because `--fill-to-memset` runs again after
      `--pack-int8-max-pool` and the row pass is no longer a `linalg` init by
      then.

    **Second attempt**, stride 1 only and the fill a memset, on GoogLeNet's nine
    remaining pools: **203.00 -> 223.78 ms, +10.2%**, and it was the only object
    that changed.

    So the operation count is not what this costs. Per output word the direct
    form is 9 loads, 8 maximums and a store -- 18; separated it is about 13. It
    is **the buffer**. A `[1, padH, outW, C]` row buffer at GoogLeNet's shapes is
    34 KB written and read back through a **16 KB L1**, where the direct form
    reads the padding buffer it already has and writes the output once. Thirteen
    operations that miss beat eighteen that hit, and they do not.

    That is the same thing [[gemmlir-the-kernel-wrote-its-own-buffer]] found from
    the other side: a kernel measures the arithmetic, and what these loops pay
    is the span they touch. **Do not revive this without a cache model**; three
    causes have been found and removed and it is still slower.
  }];
}

def HoldPerChannelInRegisters: Pass<"hold-per-channel-in-registers", "::mlir::func::FuncOp"> {
  let summary = "Strip-mine the channel so a per-channel operand is loaded once, not once an element";
  let description = [{
    **Built, tested, and deliberately not in the pipeline: exact, and worth
    nothing on the models that have the shape.**

    DenseNet's integer batch norm is `lb lw lw mul add sra bltu sb` -- and two
    of those four memory operations are `M[c]` and `N[c]`, which do not change
    as the loop walks the picture. They sit inside it because the **channel is
    the innermost dimension**: NHWC puts it there, and there is no
    loop-invariant code motion anywhere in this pipeline to lift them out
    ([[gemmlir-no-loop-invariant-motion-anywhere]]).

    So move the loop instead. Eight channels at a time become the outermost
    loop, their sixteen coefficients are loaded once into registers, and the
    picture is walked inside. Exact: the same arithmetic on the same elements in
    a different order, and every element is independent.

    A kernel at DenseNet's own shapes, against the same body unrolled the same
    eight ways, read -5.7% to -9.8% wherever the picture fit the L1 and there
    were more than four pixels, and **+16.6%** on a 32 KB picture that did not.
    Wired in, on 437 of DenseNet's 600 loops, it read:

    | | previous | held |
    |---|---|---|
    | densenet121 | 487.25 ms | 484.60 (-0.54%) |
    | mobilenet_v3_small | 109.71 | 110.26 (**+0.50%**) |
    | regnet_y_400mf | 106.71 | 106.63 (-0.07%) |
    | efficientnet_b0 | 237.38 | 237.41 (+0.01%) |
    | the set | 1735.12 | 1732.86 (-0.13%) |

    All thirteen 0 of 40 against the CPU reference and against the build before.

    **What the kernel could not see:** it wrote a buffer of its own. DenseNet's
    batch norm writes **579 of 582** of its outputs into a *strided slice of a
    concatenation* -- 32 channels of 96, so 8 KB of data spread over 24 KB of
    address space. Walking the picture once per channel group re-touches that
    whole span, and the lines it re-misses cost what the sixteen hoisted loads
    saved. A contiguous output, which is what the kernel measured, has no such
    span; restricting the pass to one leaves 3 loops and 14.3 K elements, which
    is nothing.

    **Not for a per-channel `f32`** either. EfficientNet's squeeze-and-excitation
    gate has the same shape with one runtime `f32` a channel, and holding it in
    a register is worth -1.5% to +4.4% -- its body waits on the
    convert-multiply-convert chain, not on the load.

    If it is ever revived: price the **span** the output touches, not the number
    of elements it writes, and measure a loop that writes into a concatenation.
  }];
  let options = [
    Option<"lanes", "lanes", "unsigned", /*default=*/"8",
           "Channels to hold in registers at once">,
    Option<"mostBytes", "most-bytes", "unsigned", /*default=*/"8192",
           "Largest buffer to re-read per channel group; half of this board's "
           "16 KB L1">,
    Option<"leastPixels", "least-pixels", "unsigned", /*default=*/"16",
           "Fewest pixels worth loading the coefficients for">,
  ];
}
