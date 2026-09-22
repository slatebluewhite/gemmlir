//===- GemmlirOps.td - Gemmlir dialect ops -----------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef GEMMLIR_OPS
#define GEMMLIR_OPS

include "Gemmlir/GemmlirTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dataflow
//===----------------------------------------------------------------------===//

// Values match `enum tiled_matmul_type_t {OS, WS, CPU}` in gemmini.h: the
// attribute is passed straight through as the last argument of
// tiled_matmul_auto, so the numbering is part of the ABI.
def Gemmlir_Dataflow : I32EnumAttr<"Dataflow",
    "Gemmini dataflow to request from the runtime", [
  I32EnumAttrCase<"OS",  0, "os">,
  I32EnumAttrCase<"WS",  1, "ws">,
  I32EnumAttrCase<"CPU", 2, "cpu">,
]> {
  let cppNamespace = "::mlir::gemmlir";
  let genSpecializedAttr = 0;
}

// `ws` is the default because weight-stationary is what every gemmini-rocc-tests
// workload uses and what Gemmini's own leanConfig/chipConfig support; `os` needs
// a bitstream built with Dataflow.BOTH or Dataflow.OS and is not always correct
// even then. `cpu` asks the runtime to compute on the host core; it is rejected
// by matmul_i8 because gemmini.h's matmul_cpu writes elem_t and cannot produce
// the i32 accumulators that op's result memref holds.
def Gemmlir_DataflowAttr : EnumAttr<Gemmlir_Dialect, Gemmlir_Dataflow, "dataflow"> {
  let assemblyFormat = "`<` $value `>`";
  let defaultValue = "Dataflow::WS";
}

//===----------------------------------------------------------------------===//
// Activation
//===----------------------------------------------------------------------===//

// Values match gemmini.h's NO_ACTIVATION / RELU. The accelerator applies the
// activation in the accumulator's scale pipeline on the way out, so it is only
// reachable when the result is the scaled elem_t one: with full_C the raw 32-bit
// accumulator is read back and the pipeline -- activation included -- is
// bypassed. Measured on hardware: full_C=1 with act=RELU returns the unclamped
// product for every element.
// 2, 3 and 4 need a bitstream whose Gemmini was built with `norms = true`:
// without the normalization unit the RTL asserts
// (`AccumulatorScale.scala`'s `assert(has_normalizations.B || ...)`). They are
// also not activations in the pointwise sense -- LayerNorm and Softmax reduce
// along a row -- which is why only `norm_i8` and the matmuls that end in one
// accept them; see `Conv2DInt8Op`, whose window has no rows to reduce along.
def Gemmlir_Act : I32EnumAttr<"Act", "Activation fused into the mvout", [
  I32EnumAttrCase<"NONE", 0, "none">,
  I32EnumAttrCase<"RELU", 1, "relu">,
  I32EnumAttrCase<"LAYERNORM", 2, "layernorm">,
  I32EnumAttrCase<"IGELU", 3, "igelu">,
  I32EnumAttrCase<"SOFTMAX", 4, "softmax">,
]> {
  let cppNamespace = "::mlir::gemmlir";
  let genSpecializedAttr = 0;
}

def Gemmlir_ActAttr : EnumAttr<Gemmlir_Dialect, Gemmlir_Act, "act"> {
  let assemblyFormat = "`<` $value `>`";
  let defaultValue = "Act::NONE";
}

def MemsetOp : Gemmlir_Op<"memset", [MemoryEffects<[MemWrite]>]> {
  let summary = "Write one repeated byte over a contiguous buffer";
  let description = [{
    What a `linalg.fill` of a uniform byte pattern is, once there is no reason
    to keep it as a loop. `--convert-linalg-to-gemmlir` has to see the fills --
    a zero one is how an accumulator is proved to start at zero -- so this is
    made after it, in `--fill-to-memset`.

    The buffer has to be contiguous and statically shaped, and `value` is the
    byte, not the element: an f32 fill only becomes one of these when all four
    of its bytes agree, which in practice means zero.
  }];

  let arguments = (ins
    AnyStaticShapeMemRef:$buffer,
    I8Attr:$value
  );

  let assemblyFormat = [{
    `(` $buffer `)` attr-dict `:` type($buffer)
  }];
  let hasVerifier = 1;
}

def MatMulInt8Op : Gemmlir_Op<"matmul_i8", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Tiled matmul INT8 inputs, INT32 accumulation output";
  let description = [{
    Performs A x B -> C where A,B are 2D memref<...xi8> and C is a
    2D memref<...xi32> accumulation buffer. Attributes control transposes
    and which Gemmini dataflow the runtime is asked for (`os`, `ws`, `cpu`).

    With `accumulate` (the default) the existing contents of C are added to the
    product, which is what `linalg.matmul` means; the lowering gets this by
    handing C to the runtime as the bias operand `D` as well. Set it to false
    only when C is known to be zero -- it saves reading C back into the
    accumulator.

    An explicit `bias` may be given instead: the runtime has exactly one `D`
    pointer, so `bias` and `accumulate` are mutually exclusive. Its shape says
    how it is read -- (M, N) is one value per element, (1, N) is a row broadcast
    down the matrix (`repeating_bias`).

    `lhs_scale` and `rhs_scale` are applied as the operands are loaded: the
    runtime's `MVIN_SCALE` is `round_nearest_even(x * scale)` **saturated back to
    i8**, so this requantizes an input rather than scaling the arithmetic. Leave
    them at 1.0 for an exact integer product.

    No activation: this op reads the raw 32-bit accumulators, which bypasses the
    pipeline the accelerator applies activations in. Use `matmul_i8_scale` for a
    fused activation.
  }];

  let arguments = (ins
    MemRefRankOf<[I8], [2]>:$lhsMat,
    MemRefRankOf<[I8], [2]>:$rhsMat,
    MemRefRankOf<[I32], [2]>:$outMat,
    Optional<MemRefRankOf<[I32], [2]>>:$bias,
    DefaultValuedAttr<F32Attr, "1.0f">:$lhs_scale,
    DefaultValuedAttr<F32Attr, "1.0f">:$rhs_scale,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs,
    DefaultValuedAttr<BoolAttr, "true">:$accumulate,
    DefaultValuedAttr<Gemmlir_DataflowAttr, "Dataflow::WS">:$dataflow
  );

  let assemblyFormat = [{
    `(` $lhsMat `,` $rhsMat `,` $outMat `)` (`bias` `(` $bias^ `:` type($bias) `)`)?
    `:` `(` type($lhsMat) `x` type($rhsMat) `)` `->` type($outMat) attr-dict
  }];

  let hasVerifier = 1;
}

def MatMulInt8ScaleOp : Gemmlir_Op<"matmul_i8_scale", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Tiled matmul INT8 inputs, INT8 scaled output";
  let description = [{
    Performs `C = act(scale * (lhs_scale*A x rhs_scale*B))` where A, B and C are
    2D memref<...xi8>. The accumulator is scaled by `scale` and saturated back to
    i8 on the way out, and `act` is applied in the same pipeline.

    `lhs_scale` / `rhs_scale` are the operands' mvin scales, applied on load and
    saturated back to i8 -- an input requantization, as on `matmul_i8`.

    Unlike `matmul_i8` this op cannot accumulate -- C is i8 while the runtime's
    bias operand is i32 -- but it can take that i32 `bias`, which is the usual
    shape of a quantized matmul. (M, N) is one value per element, (1, N) a row
    broadcast down the matrix.
  }];

  let arguments = (ins
    MemRefRankOf<[I8], [2]>:$lhsMat,
    MemRefRankOf<[I8], [2]>:$rhsMat,
    MemRefRankOf<[I8], [2]>:$outMat,
    Optional<MemRefRankOf<[I32], [2]>>:$bias,
    DefaultValuedAttr<F32Attr, "1.0f">:$lhs_scale,
    DefaultValuedAttr<F32Attr, "1.0f">:$rhs_scale,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs,
    DefaultValuedAttr<F32Attr, "1.0f">:$scale,
    // What one unit of the accumulator is worth, for the activations that need
    // to know. `igelu` and `softmax` are the I-BERT integer approximations and
    // their constants are derived from it: the runtime computes `qln2`, `qb`
    // and `qc` out of this and configures the normalizer with them. For a
    // quantized matmul the right value is the accumulator's own scale,
    // `lhs_scale * rhs_scale` -- give it anything else and the softmax comes
    // out at the wrong temperature. `relu` and `none` never read it.
    DefaultValuedAttr<F32Attr, "1.0f">:$bert_scale,
    DefaultValuedAttr<Gemmlir_ActAttr, "Act::NONE">:$act,
    DefaultValuedAttr<Gemmlir_DataflowAttr, "Dataflow::WS">:$dataflow
  );

  let assemblyFormat = [{
    `(` $lhsMat `,` $rhsMat `,` $outMat `)` (`bias` `(` $bias^ `:` type($bias) `)`)?
    `:` `(` type($lhsMat) `x` type($rhsMat) `)` `->` type($outMat) attr-dict
  }];

  let hasVerifier = 1;
}

def ResAddInt8Op : Gemmlir_Op<"resadd_i8", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Elementwise C = act(c_scale * (a_scale*A + b_scale*B)) on i8";
  let description = [{
    Lowers to `tiled_resadd_auto`. A, B and C are 2D memref<...xi8> of the same
    shape; the runtime assumes rows are contiguous, so the shapes must match
    exactly rather than merely agree in element count.
  }];

  let arguments = (ins
    MemRefRankOf<[I8], [2]>:$lhsMat,
    MemRefRankOf<[I8], [2]>:$rhsMat,
    MemRefRankOf<[I8], [2]>:$outMat,
    DefaultValuedAttr<F32Attr, "1.0f">:$lhs_scale,
    DefaultValuedAttr<F32Attr, "1.0f">:$rhs_scale,
    DefaultValuedAttr<F32Attr, "1.0f">:$out_scale,
    DefaultValuedAttr<Gemmlir_ActAttr, "Act::NONE">:$act,
    DefaultValuedAttr<Gemmlir_DataflowAttr, "Dataflow::WS">:$dataflow
  );

  let assemblyFormat = [{
    `(` $lhsMat `,` $rhsMat `,` $outMat `)` `:` `(` type($lhsMat) `x` type($rhsMat) `)` `->` type($outMat) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Convolution
//===----------------------------------------------------------------------===//
//
// Layouts are the runtime's, which for the dense case happen to be exactly
// linalg's NHWC/HWCF: conv_cpu indexes input as
// `[(b*H*W + irow*W + icol)*C + kch]` and weights as
// `[(krow*K*C + kcol*C + kch)*F + och]`. The depthwise kernel does not follow
// linalg -- it is `[(ch*K + krow)*K + kcol]`, i.e. (C, KH, KW).

def NormInt8Op : Gemmlir_Op<"norm_i8", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Row-wise normalization of an i32 accumulator, scaled to i8";
  let description = [{
    `output[i][j] = sat_i8(round(scale * norm(input[i])[j]))`, lowered to
    `tiled_norm_auto`.

    The normalization runs along a row, so `input` is the i32 accumulator a
    matmul left behind and `output` is the i8 the next layer reads. Three of
    them:

    * `layernorm` subtracts each row's mean and divides by its standard
      deviation, both computed in integers.
    * `softmax` is the I-BERT integer approximation -- a second-order polynomial
      for `exp` on the reduced range, after subtracting the row maximum. It is
      not IEEE softmax and does not claim to be; the runtime's own CPU path
      computes the same approximation, which is what it is checked against.
    Not iGELU, although the hardware has it: `sp_tiled_norm` branches on
    LAYERNORM and SOFTMAX and has no third case, so an iGELU mvins the
    accumulator and never mvouts -- measured on the board, every element of the
    output left as it was. It is reachable on the matmul's own scale pipeline,
    which does configure its constants.

    The constants softmax needs (`qln2`, `qb`, `qc`) come from a BERT scale the
    runtime hard-codes at 0.05, so the approximation does not follow the data's
    scale. There is nothing to pass and nothing to choose.

    Needs a bitstream with the normalization unit. Without it the whole thing
    compiles to nothing: `sp_tiled_norm`'s body is behind `HAS_NORMALIZATIONS`,
    so the call returns having written no output at all.
  }];

  let arguments = (ins
    MemRefRankOf<[I32], [2]>:$input,
    MemRefRankOf<[I8], [2]>:$output,
    DefaultValuedAttr<F32Attr, "1.0f">:$scale,
    // No default: the three are not variations on one thing, and a
    // normalization that does not say which it is has not said anything.
    Gemmlir_ActAttr:$act
  );

  let assemblyFormat = [{
    `(` $input `,` $output `)` attr-dict `:` `(` type($input) `,` type($output) `)`
  }];
  let hasVerifier = 1;
}

//
// Only square kernels, strides, paddings and dilations are expressible: the
// runtime takes one int for each. `wrot180` and the four transpose flags exist
// in the ABI but are pinned to false here.

class Gemmlir_ConvBase<string mnemonic> :
    Gemmlir_Op<mnemonic, [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;
  // `bias` is a labelled clause rather than a third positional operand: with the
  // optional group inline the parser cannot tell, after the filter, whether the
  // next operand is the bias or the output.
  let assemblyFormat = [{
    `(` $input `,` $filter `,` $output `)` (`bias` `(` $bias^ `:` type($bias) `)`)?
    attr-dict `:` `(` type($input) `,` type($filter) `,` type($output) `)`
  }];
}

def Conv2DInt8Op : Gemmlir_ConvBase<"conv2d_i8"> {
  let summary = "Tiled int8 2-D convolution with fused bias, activation and pooling";
  let description = [{
    `output = pool(act(scale * (conv(input, filter) + bias)))`, lowered to
    `tiled_conv_auto`.

    `input` is NHWC, `filter` is (KH, KW, C, F) with KH == KW, `bias` is one i32
    per output channel, `output` is NHWC and i8 -- the accumulator is scaled and
    saturated on the way out, like `matmul_i8_scale`.

    The convolution's own output extent is
    `(H + 2*padding - dilation*(K-1) - 1)/stride + 1`. With `pool_stride` zero
    that is the shape of `output`; otherwise max-pooling is fused and `output`
    holds `(OH + 2*pool_padding - pool_size)/pool_stride + 1` instead.

    `input_dilation` spreads the *input* instead of the filter: the convolution
    reads as if `input_dilation - 1` zeros sat between every pair of samples,
    which is what a transposed convolution is. The input extent above becomes
    `input_dilation*(H-1) + 1`. The accelerator takes 1 or 2 and only with
    `stride` 1; anything else is a CPU fallback in the runtime that exits.
  }];

  let arguments = (ins
    MemRefRankOf<[I8], [4]>:$input,
    MemRefRankOf<[I8], [4]>:$filter,
    Optional<MemRefRankOf<[I32], [1]>>:$bias,
    MemRefRankOf<[I8], [4]>:$output,
    DefaultValuedAttr<I64Attr, "1">:$stride,
    DefaultValuedAttr<I64Attr, "0">:$padding,
    DefaultValuedAttr<I64Attr, "1">:$dilation,
    DefaultValuedAttr<I64Attr, "1">:$input_dilation,
    DefaultValuedAttr<F32Attr, "1.0f">:$scale,
    DefaultValuedAttr<Gemmlir_ActAttr, "Act::NONE">:$act,
    DefaultValuedAttr<I64Attr, "0">:$pool_size,
    DefaultValuedAttr<I64Attr, "0">:$pool_stride,
    DefaultValuedAttr<I64Attr, "0">:$pool_padding,
    DefaultValuedAttr<Gemmlir_DataflowAttr, "Dataflow::WS">:$dataflow
  );
}

def DepthwiseConv2DInt8Op : Gemmlir_ConvBase<"depthwise_conv2d_i8"> {
  let summary = "Tiled int8 depthwise 2-D convolution";
  let description = [{
    As `conv2d_i8`, one filter per channel, lowered to `tiled_conv_dw_auto`.

    `filter` is (C, KH, KW) with KH == KW -- channel-major, which is *not*
    linalg's (KH, KW, C). The runtime has no dilation for this kernel.
  }];

  let arguments = (ins
    MemRefRankOf<[I8], [4]>:$input,
    MemRefRankOf<[I8], [3]>:$filter,
    Optional<MemRefRankOf<[I32], [1]>>:$bias,
    MemRefRankOf<[I8], [4]>:$output,
    DefaultValuedAttr<I64Attr, "1">:$stride,
    DefaultValuedAttr<I64Attr, "0">:$padding,
    DefaultValuedAttr<F32Attr, "1.0f">:$scale,
    DefaultValuedAttr<Gemmlir_ActAttr, "Act::NONE">:$act,
    DefaultValuedAttr<I64Attr, "0">:$pool_size,
    DefaultValuedAttr<I64Attr, "0">:$pool_stride,
    DefaultValuedAttr<I64Attr, "0">:$pool_padding,
    DefaultValuedAttr<Gemmlir_DataflowAttr, "Dataflow::WS">:$dataflow
  );
}

#endif // GEMMLIR_OPS
