//===- GemmlirOps.h - Gemmlir dialect ops -----------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef GEMMLIR_GEMMLIROPS_H
#define GEMMLIR_GEMMLIROPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Gemmlir/GemmlirAttrs.h"

#define GET_OP_CLASSES
#include "Gemmlir/GemmlirOps.h.inc"

namespace mlir {
namespace gemmlir {

/// The logical M x K x N of a matmul plus the row strides the runtime needs,
/// derived from how the operands are actually stored.
struct MatmulShape {
  int64_t M, K, N;
  int64_t strideA, strideB, strideC;
};

/// Resolves `lhs x rhs -> out` under the given transposes.
///
/// `tiled_matmul_auto` reads A[i][k] at `A + i*sA + k` when A is stored plainly
/// and at `A + i + k*sA` when it is transposed, and B[k][j] at `B + j + k*sB`
/// / `B + j*sB + k` -- so each stride is just the trailing dimension of the
/// memref as stored, and the transpose flags decide which axis is which.
///
/// Returns nullopt when the shapes do not agree; the caller reports why.
std::optional<MatmulShape> computeMatmulShape(MemRefType lhs, MemRefType rhs,
                                              MemRefType out, bool transposeLhs,
                                              bool transposeRhs);

/// Row stride of a 2-D memref the runtime can address, or nullopt when the
/// layout is not row-major with unit-stride columns.
std::optional<int64_t> rowStrideOf(MemRefType type);

} // namespace gemmlir
} // namespace mlir

#endif // GEMMLIR_GEMMLIROPS_H
