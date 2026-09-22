//===- RoundQuantizedCastsPass.cpp -------------------------*- C++ -*-===//
//
// Restores round-to-nearest and saturation in the quantization the quant
// dialect lowers to.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_ROUNDQUANTIZEDCASTS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// `--lower-quant-ops` turns a `quant.qcast` into `fptosi(divf(x, scale))`.
///
/// Two things are wrong with that as a quantization.
///
/// `arith.fptosi` truncates toward zero, and quantization is defined as
/// round-to-nearest; the difference is not small -- on a PyTorch MLP it was the
/// whole gap between a relative L2 error of 0.036 and 0.012.
///
/// It also does not saturate. An activation outside the calibrated range is
/// undefined behaviour in `arith.fptosi` and *wraps* on RISC-V, so a value just
/// past +127 comes back as a large negative number: not a slightly worse answer
/// but a broken one, and it fails silently the moment an input is bigger than
/// whatever the calibration run happened to see. Every definition of int8
/// quantization saturates instead, and so does the accelerator: `gemmini.h`
/// scales the i32 accumulator and clips it to `elem_t`. This emits that,
/// converting to i32 first and clamping there, which is the same arithmetic the
/// hardware does -- and it is also the shape `--convert-linalg-to-gemmlir`
/// recognises as a requantization.
///
/// Only that exact shape is rewritten -- a float divide feeding an integer
/// conversion -- so a truncation the input asked for on its own is left alone.
class RoundBeforeConversion : public OpRewritePattern<arith::FPToSIOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::FPToSIOp op,
                                PatternRewriter &rewriter) const final {
    Value in = op.getIn();
    auto round = in.getDefiningOp<math::RoundEvenOp>();
    Value beforeRound = round ? round.getOperand() : in;
    if (!beforeRound.getDefiningOp<arith::DivFOp>())
      return failure();

    // The conversion runs on tensors here (this pass sits between
    // --lower-quant-ops and --convert-elementwise-to-linalg), so the storage
    // width is the *element* type's.
    Type resTy = op.getType();
    auto narrow = llvm::dyn_cast<IntegerType>(getElementTypeOrSelf(resTy));
    bool needsClamp = narrow && narrow.getWidth() < 32;
    // Already rounded and wide enough not to need clamping: nothing to do.
    // This is also what stops the rewrite below from matching its own output,
    // whose conversion is to i32 over a `roundeven`.
    if (round && !needsClamp)
      return failure();

    Location loc = op.getLoc();
    Value rounded =
        round ? in : rewriter.create<math::RoundEvenOp>(loc, beforeRound).getResult();
    if (!needsClamp) {
      rewriter.modifyOpInPlace(op, [&] { op.getInMutable().assign(rounded); });
      return success();
    }

    // Convert wide, clamp to the storage type's range, then narrow.
    auto i32 = rewriter.getI32Type();
    Type wideTy = llvm::isa<ShapedType>(resTy)
                      ? llvm::cast<ShapedType>(resTy).clone(i32)
                      : llvm::cast<Type>(i32);
    Value wide = rewriter.create<arith::FPToSIOp>(loc, wideTy, rounded);
    auto bound = [&](int64_t v) -> Value {
      auto attr = rewriter.getI32IntegerAttr(static_cast<int32_t>(v));
      if (auto shaped = llvm::dyn_cast<ShapedType>(wideTy))
        return rewriter.create<arith::ConstantOp>(
            loc, wideTy, DenseElementsAttr::get(shaped, attr.getValue()));
      return rewriter.create<arith::ConstantOp>(loc, wideTy, attr);
    };
    unsigned bits = narrow.getWidth();
    Value clamped = rewriter.create<arith::MinSIOp>(
        loc, rewriter.create<arith::MaxSIOp>(loc, wide, bound(llvm::minIntN(bits))),
        bound(llvm::maxIntN(bits)));
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, resTy, clamped);
    return success();
  }
};

class RoundQuantizedCasts
    : public impl::RoundQuantizedCastsBase<RoundQuantizedCasts> {
public:
  using impl::RoundQuantizedCastsBase<RoundQuantizedCasts>::RoundQuantizedCastsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RoundBeforeConversion>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
