//===- QuantizeBiasIntoAccumulatorPass.cpp -----------------*- C++ -*-===//
//
// Moves a constant bias from the dequantized domain into the integer one.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cmath>

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_QUANTIZEBIASINTOACCUMULATOR
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// `acc * s + b` becomes `(acc + round(b/s)) * s`.
///
/// The accelerator's bias operand is added to the i32 accumulator before the
/// mvout scaling, so a bias added in f32 afterwards -- which is how a frontend
/// writes it -- cannot reach it, and the whole tail of the layer stays in
/// software. Moving a *constant* bias inside is the standard int8 bias
/// quantization: `round(b/s)` is evaluated here, and the error is half a step of
/// `s`, the accumulator's own resolution.
///
/// The rewrite is inside one `linalg.generic` body, so it does not care how many
/// reshapes the frontend put between the contraction and the bias.
class QuantizeBias : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getOutputs().size() != 1)
      return failure();
    // Only where the result is requantized back to an integer -- which is what
    // the accelerator's mvout does. When the layer's result stays in float
    // there is nothing to fold the bias into, and adding it in f32 is strictly
    // more accurate, so the last layer of a network keeps its own bias.
    if (!getElementTypeOrSelf(generic.getResult(0).getType()).isInteger())
      return failure();
    Block &body = generic.getRegion().front();

    // Find `addf(mulf(sitofp(%acc), s), %bias)` with both %acc and %bias
    // operands of this operation.
    arith::AddFOp add;
    arith::MulFOp mul;
    arith::SIToFPOp toFloat;
    BlockArgument accArg, biasArg;
    llvm::APFloat scale(0.0f);

    for (arith::AddFOp candidate : body.getOps<arith::AddFOp>()) {
      for (unsigned side = 0; side < 2; side++) {
        auto m = candidate.getOperand(side).getDefiningOp<arith::MulFOp>();
        auto b = llvm::dyn_cast<BlockArgument>(candidate.getOperand(1 - side));
        if (!m || !b || b.getOwner() != &body)
          continue;
        Value scaled;
        llvm::APFloat s(0.0f);
        if (matchPattern(m.getRhs(), m_ConstantFloat(&s)))
          scaled = m.getLhs();
        else if (matchPattern(m.getLhs(), m_ConstantFloat(&s)))
          scaled = m.getRhs();
        else
          continue;
        auto f = scaled.getDefiningOp<arith::SIToFPOp>();
        if (!f)
          continue;
        auto a = llvm::dyn_cast<BlockArgument>(f.getIn());
        if (!a || a.getOwner() != &body || !a.getType().isInteger(32))
          continue;
        // Every value being rerouted must have exactly the one use, or moving
        // the bias inside would change what something else reads.
        if (!a.hasOneUse() || !b.hasOneUse() || !m->hasOneUse() ||
            !f->hasOneUse())
          continue;
        add = candidate; mul = m; toFloat = f; accArg = a; biasArg = b;
        scale = s;
        break;
      }
      if (add)
        break;
    }
    if (!add)
      return failure();

    double s = scale.convertToFloat();
    if (!(s > 0.0))
      return failure();

    // The bias operand has to be a constant to be evaluated here.
    unsigned biasIdx = biasArg.getArgNumber();
    if (biasIdx >= generic.getInputs().size())
      return failure();

    // Only when the accelerator could actually take the result. Its `D` is
    // either a full tile or a single row repeated down the rows, so a bias
    // broadcast over anything but the trailing dimension can never reach it --
    // which is the shape an img2col'd convolution has, where the channels are
    // the *rows*. Quantizing it there would cost accuracy and buy nothing: on
    // the NCHW CNN it moved the relative L2 from 0.0054 to 0.0067 with not one
    // extra operation offloaded.
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    AffineMap biasMap = maps[biasIdx];
    unsigned rank = maps.back().getNumDims();
    MLIRContext *ctx = rewriter.getContext();
    bool repeatingRow =
        biasMap == AffineMap::get(rank, 0, {getAffineDimExpr(rank - 1, ctx)}, ctx);
    if (!repeatingRow && !(rank == 2 && biasMap.isIdentity()))
      return failure();
    Value biasVal = generic.getInputs()[biasIdx];
    auto cst = biasVal.getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return failure();
    auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
    auto biasTy = llvm::dyn_cast<RankedTensorType>(biasVal.getType());
    if (!dense || !biasTy || !biasTy.getElementType().isF32())
      return failure();

    // Round to nearest, and refuse anything that would crowd the accumulator:
    // this value is added to a sum the hardware keeps in 32 bits.
    SmallVector<llvm::APInt> quantized;
    quantized.reserve(dense.getNumElements());
    for (const llvm::APFloat &f : dense.getValues<llvm::APFloat>()) {
      double q = std::nearbyint(static_cast<double>(f.convertToFloat()) / s);
      if (std::abs(q) > (double)(1 << 30))
        return failure();
      quantized.push_back(llvm::APInt(32, (int64_t)q, /*isSigned=*/true));
    }
    auto i32 = rewriter.getI32Type();
    auto biasI32Ty = RankedTensorType::get(biasTy.getShape(), i32);

    Location loc = generic.getLoc();
    SmallVector<Value> inputs(generic.getInputs());
    inputs[biasIdx] = rewriter.create<arith::ConstantOp>(
        loc, biasI32Ty, DenseElementsAttr::get(biasI32Ty, quantized));

    auto rewritten = rewriter.create<linalg::GenericOp>(
        loc, generic.getResultTypes(), inputs, generic.getOutputs(),
        generic.getIndexingMapsArray(), generic.getIteratorTypesArray());

    // Clone the body with the accumulator argument standing for `acc + bias`,
    // which makes the existing sitofp read the biased value, and with the
    // float add replaced by the multiply it used to wrap.
    Block *newBody = rewriter.createBlock(
        &rewritten.getRegion(), rewritten.getRegion().begin(),
        body.getArgumentTypes(), llvm::to_vector(llvm::map_range(
            body.getArguments(), [&](BlockArgument a) { return a.getLoc(); })));
    newBody->getArgument(biasIdx).setType(i32);

    IRMapping map;
    rewriter.setInsertionPointToStart(newBody);
    Value biased = rewriter.create<arith::AddIOp>(
        loc, newBody->getArgument(accArg.getArgNumber()),
        newBody->getArgument(biasIdx));
    for (auto [oldArg, newArg] :
         llvm::zip(body.getArguments(), newBody->getArguments()))
      map.map(oldArg, newArg);
    map.map(accArg, biased);

    for (Operation &op : body.without_terminator()) {
      if (&op == add.getOperation()) {
        map.map(add.getResult(), map.lookup(mul.getResult()));
        continue;
      }
      rewriter.clone(op, map);
    }
    auto yield = llvm::cast<linalg::YieldOp>(body.getTerminator());
    SmallVector<Value> results;
    for (Value v : yield.getOperands())
      results.push_back(map.lookupOrDefault(v));
    rewriter.create<linalg::YieldOp>(loc, results);

    rewriter.replaceOp(generic, rewritten.getResults());
    return success();
  }
};

class QuantizeBiasIntoAccumulator
    : public impl::QuantizeBiasIntoAccumulatorBase<QuantizeBiasIntoAccumulator> {
public:
  using impl::QuantizeBiasIntoAccumulatorBase<
      QuantizeBiasIntoAccumulator>::QuantizeBiasIntoAccumulatorBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<QuantizeBias>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
