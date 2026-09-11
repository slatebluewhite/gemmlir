//===- GemmlirOps.td - Gemmlir dialect ops -----------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef GEMMLIR_OPS
#define GEMMLIR_OPS

include "Gemmlir/GemmlirTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MatMulInt8Op : Gemmlir_Op<"matmul_i8", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Tiled matmul INT8 inputs, INT32 accumulation output";
  let description = [{
    Performs A x B -> C where A,B are 2D memref<...xi8> and C is
    2D memref<...xi32> accumulation buffer. Attributes control transposes.
  }];

  let arguments = (ins
    MemRefRankOf<[I8], [2]>:$lhsMat,
    MemRefRankOf<[I8], [2]>:$rhsMat,
    MemRefRankOf<[I32], [2]>:$outMat,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs
  );

  let assemblyFormat = [{
    `(` $lhsMat `,` $rhsMat `,` $outMat `)` `:` `(` type($lhsMat) `x` type($rhsMat) `)` `->` type($outMat) attr-dict
  }];
}

def MatMulInt8ScaleOp : Gemmlir_Op<"matmul_i8_scale", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Tiled matmul INT8 inputs, INT8 scaled output";
  let description = [{
    Performs A x B -> C where A,B,C are 2D memref<...xi8>. Attributes
    control transposes. Intended for post-accumulation scaled output.
  }];

  let arguments = (ins
    MemRefRankOf<[I8], [2]>:$lhsMat,
    MemRefRankOf<[I8], [2]>:$rhsMat,
    MemRefRankOf<[I8], [2]>:$outMat,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs
  );

  let assemblyFormat = [{
    `(` $lhsMat `,` $rhsMat `,` $outMat `)` `:` `(` type($lhsMat) `x` type($rhsMat) `)` `->` type($outMat) attr-dict
  }];
}

#endif // GEMMLIR_OPS
