//===- GemmlirTypes.cpp - Gemmlir dialect types -----------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "Gemmlir/GemmlirTypes.h"

#include "Gemmlir/GemmlirDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::gemmlir;

#define GET_TYPEDEF_CLASSES
#include "Gemmlir/GemmlirOpsTypes.cpp.inc"

void GemmlirDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Gemmlir/GemmlirOpsTypes.cpp.inc"
      >();
}
