//===- GemmlirPatterns.h -----------------------------------*- C++ -*-===//
//
// Rewrites that more than one gemmlir pass needs.
//
//===----------------------------------------------------------------------===//

#ifndef GEMMLIR_GEMMLIRPATTERNS_H
#define GEMMLIR_GEMMLIRPATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::gemmlir {

/// `elementwise(transpose(x))` is `elementwise(x)` read through the
/// permutation.
///
/// The layout rewrite needs this to walk the transposes it creates into the
/// operations between two layers, where they cancel. The elementwise fusion
/// needs it again afterwards, because the quantization of a network's input is
/// only created once the quantization passes have run, and by then it sits on
/// top of the one transpose the layout rewrite could not cancel -- folding the
/// two saves a whole pass over the data.
void populateAbsorbTransposePatterns(RewritePatternSet &patterns);

} // namespace mlir::gemmlir

#endif // GEMMLIR_GEMMLIRPATTERNS_H
