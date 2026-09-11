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

#define GET_OP_CLASSES
#include "Gemmlir/GemmlirOps.h.inc"

#endif // GEMMLIR_GEMMLIROPS_H
