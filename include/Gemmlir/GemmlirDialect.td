//===- GemmlirDialect.td - Gemmlir dialect -----------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef GEMMLIR_DIALECT
#define GEMMLIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Gemmlir dialect definition.
//===----------------------------------------------------------------------===//

def Gemmlir_Dialect : Dialect {
    let name = "gemmlir";
    let summary = "Gemmini accelerator offload dialect.";
    let description = [{
        Ops that represent work handed to the Gemmini systolic-array
        accelerator. They are produced from linalg.matmul and lowered to calls
        into the Gemmini runtime library (gemmini.h).
    }];
    let cppNamespace = "::mlir::gemmlir";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base gemmlir operation definition.
//===----------------------------------------------------------------------===//

class Gemmlir_Op<string mnemonic, list<Trait> traits = []> :
        Op<Gemmlir_Dialect, mnemonic, traits>;

#endif // GEMMLIR_DIALECT
