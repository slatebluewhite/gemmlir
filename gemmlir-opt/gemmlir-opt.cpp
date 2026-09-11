//===- gemmlir-opt.cpp ---------------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Gemmlir/GemmlirDialect.h"
#include "Gemmlir/GemmlirPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::gemmlir::registerPasses();

  mlir::DialectRegistry registry;
  // Register all core MLIR dialects and external dialect extensions (e.g.
  // BufferizableOpInterface implementations for arith/tensor/etc.). This is
  // required for One-Shot Bufferize.
  mlir::registerAllDialects(registry);
  // Also ensure our custom dialect is registered.
  registry.insert<mlir::gemmlir::GemmlirDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Gemmlir optimizer driver\n", registry));
}
