//===- GemmlirAttrs.h - Gemmlir dialect attributes --------*- C++ -*-===//
//
// The `dataflow` enum attribute carried by the gemmlir matmul ops. Its
// numbering mirrors `enum tiled_matmul_type_t` in gemmini.h.
//
//===----------------------------------------------------------------------===//

#ifndef GEMMLIR_GEMMLIRATTRS_H
#define GEMMLIR_GEMMLIRATTRS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

#include "Gemmlir/GemmlirEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Gemmlir/GemmlirAttrs.h.inc"

#endif // GEMMLIR_GEMMLIRATTRS_H
