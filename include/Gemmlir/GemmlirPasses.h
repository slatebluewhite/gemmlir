//===- GemmlirPasses.h - Gemmlir passes  ------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
#ifndef GEMMLIR_GEMMLIRPASSES_H
#define GEMMLIR_GEMMLIRPASSES_H

#include "Gemmlir/GemmlirDialect.h"
#include "Gemmlir/GemmlirOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace gemmlir {
#define GEN_PASS_DECL
#include "Gemmlir/GemmlirPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Gemmlir/GemmlirPasses.h.inc"
} // namespace gemmlir
} // namespace mlir

#endif
