//===- GemmlirDialect.cpp - Gemmlir dialect ---------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "Gemmlir/GemmlirDialect.h"
#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirTypes.h"

using namespace mlir;
using namespace mlir::gemmlir;

#include "Gemmlir/GemmlirOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Gemmlir dialect.
//===----------------------------------------------------------------------===//

void GemmlirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Gemmlir/GemmlirOps.cpp.inc"
      >();
  registerTypes();
}
