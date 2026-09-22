//===- GemmlirDialect.cpp - Gemmlir dialect ---------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "Gemmlir/GemmlirDialect.h"
#include "Gemmlir/GemmlirAttrs.h"
#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::gemmlir;

#include "Gemmlir/GemmlirOpsDialect.cpp.inc"
#include "Gemmlir/GemmlirEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Gemmlir/GemmlirAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Gemmlir dialect.
//===----------------------------------------------------------------------===//

void GemmlirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Gemmlir/GemmlirOps.cpp.inc"
      >();
  registerTypes();
  registerAttributes();
}

void GemmlirDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Gemmlir/GemmlirAttrs.cpp.inc"
      >();
}
