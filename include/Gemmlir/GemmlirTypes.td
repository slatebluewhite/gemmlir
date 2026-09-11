//===- GemmlirTypes.td - Gemmlir dialect types -------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef GEMMLIR_TYPES
#define GEMMLIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Gemmlir/GemmlirDialect.td"

//===----------------------------------------------------------------------===//
// Gemmlir type definitions
//===----------------------------------------------------------------------===//

class Gemmlir_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Gemmlir_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Gemmlir_CustomType : Gemmlir_Type<"Custom", "custom"> {
    let summary = "Gemmlir custom type";
    let description = "Custom type in gemmlir dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // GEMMLIR_TYPES
