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
  }];
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
    and dequant operations. This is a placeholder; implementation will follow.
  }];
}

#endif // GEMMLIR_PASS
