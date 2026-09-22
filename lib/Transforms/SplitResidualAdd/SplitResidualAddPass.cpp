//===- SplitResidualAddPass.cpp ------------------------------*- C++ -*-===//
//
// Separates a residual add from the requantizations it arrives fused with.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_SPLITRESIDUALADD
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The saturating narrowing a quantization ends in.
Value skipClamp(Value v) {
  while (true) {
    if (auto min = v.getDefiningOp<arith::MinSIOp>()) {
      v = min.getLhs();
      continue;
    }
    if (auto max = v.getDefiningOp<arith::MaxSIOp>()) {
      v = max.getLhs();
      continue;
    }
    return v;
  }
}

bool isConstantFloat(Value v, double *out) {
  llvm::APFloat f(0.0f);
  if (!matchPattern(v, m_ConstantFloat(&f)))
    return false;
  *out = f.convertToDouble();
  return true;
}

bool isContraction(Operation *op) {
  return llvm::isa_and_nonnull<linalg::Conv2DNhwcHwcfOp, linalg::MatmulOp,
                               linalg::BatchMatmulOp,
                               linalg::DepthwiseConv2DNhwcHwcOp>(op);
}

/// One side of the add: an operand of the generic, scaled into floating point.
///
/// A shortcut that is already quantized arrives as i8 and only needs scaling. A
/// branch that is still an accumulator arrives as i32, with the contraction
/// that produced it above and possibly a bias to add first -- that is the one
/// this pass takes apart.
struct Branch {
  unsigned input = 0;             // which operand of the generic
  int bias = -1;                  // and which is its bias, if any
  arith::SIToFPOp toFloat;        // sitofp(acc) or sitofp(acc + bias)
  arith::MulFOp scaled;           // its one use: a multiply by the scale
  arith::AddIOp withBias;         // null when there is no bias
  double scale = 0.0;
  bool isAccumulator = false;
};

/// In a residual block the convolutions' tails *are* the add. With an identity
/// shortcut that is one accumulator beside an i8 tensor; where the block
/// changes shape -- every stage transition of every ResNet -- the shortcut is a
/// 1x1 projection and **both** sides are accumulators. Either way
/// `--convert-linalg-to-gemmlir` reads a requantization of exactly one
/// accumulator and cannot fold it, so the convolutions stay scalar loops.
///
/// This puts each convolution's own requantization back into its own operation,
/// where the conversion finds it, and leaves a scaled add of two i8 tensors,
/// which is what `gemmlir.resadd_i8` is.
///
/// The intermediates are quantized at the block output's scale. That makes the
/// add's own arithmetic exact, but it rounds and clips each convolution's
/// result to i8 where the fused form kept it in i32 -- the approximation this
/// pass trades the offload for.
class SplitTail : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    size_t nIn = generic.getInputs().size();
    if (nIn < 2 || nIn > 4 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != nIn + 1 || !maps.back().isIdentity())
      return failure();

    auto outTy = dyn_cast<RankedTensorType>(generic.getOutputs()[0].getType());
    if (!outTy || !outTy.hasStaticShape() || !outTy.getElementType().isInteger(8))
      return failure();

    SmallVector<Branch> branches;
    double outScale = 0.0;
    bool relu = false;
    if (!matchResidual(generic, &branches, &outScale, &relu))
      return failure();

    // Only worth doing where there is an accelerator call to uncover.
    bool any = false;
    for (Branch &b : branches) {
      if (!b.isAccumulator)
        continue;
      if (!isContraction(generic.getInputs()[b.input].getDefiningOp()))
        return failure();
      any = true;
    }
    if (!any)
      return failure();

    Location loc = generic.getLoc();
    auto f32 = rewriter.getF32Type();
    // The intermediates carry the block output's scale, so the add below needs
    // no rescaling of them at all.
    double tScale = outScale;

    // Each accumulator gets its own requantization, in the shape
    // --convert-linalg-to-gemmlir reads: the accumulator first, then the bias.
    SmallVector<Value> addIns(branches.size());
    SmallVector<AffineMap> addMaps;
    IRMapping bodyMap;
    SmallVector<Value> replacedScales;
    for (Branch &b : branches) {
      if (!b.isAccumulator) {
        addIns[&b - branches.begin()] = generic.getInputs()[b.input];
        addMaps.push_back(maps[b.input]);
        replacedScales.push_back(Value());
        continue;
      }
      SmallVector<Value> reqIns = {generic.getInputs()[b.input]};
      SmallVector<AffineMap> reqMaps = {maps[b.input]};
      if (b.bias >= 0) {
        reqIns.push_back(generic.getInputs()[b.bias]);
        reqMaps.push_back(maps[b.bias]);
      }
      reqMaps.push_back(maps.back());
      Value init = rewriter.create<tensor::EmptyOp>(loc, outTy.getShape(),
                                                    outTy.getElementType());
      bool hasBias = b.bias >= 0;
      double factor = b.scale / tScale;
      auto requant = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{outTy}, reqIns, ValueRange{init}, reqMaps,
          generic.getIteratorTypesArray(),
          [&](OpBuilder &nested, Location l, ValueRange args) {
            Value acc = args[0];
            if (hasBias)
              acc = nested.create<arith::AddIOp>(l, acc, args[1]);
            Value wide = nested.create<arith::SIToFPOp>(l, f32, acc);
            Value scaled = nested.create<arith::MulFOp>(
                l, wide,
                nested.create<arith::ConstantOp>(l, nested.getF32FloatAttr(factor)));
            nested.create<linalg::YieldOp>(l, narrow(nested, l, scaled));
          });
      addIns[&b - branches.begin()] = requant.getResult(0);
      addMaps.push_back(maps.back());
      replacedScales.push_back(requant.getResult(0));
    }
    addMaps.push_back(maps.back());

    // The add: the original body with each accumulator's scaled value replaced
    // by its intermediate's, so the activation and the output scaling are kept
    // exactly as they were written.
    Block &old = generic.getRegion().front();
    auto added = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{outTy}, addIns, generic.getOutputs(), addMaps,
        generic.getIteratorTypesArray(),
        [&](OpBuilder &nested, Location l, ValueRange args) {
          IRMapping map;
          SmallVector<Operation *> skip;
          for (auto [i, b] : llvm::enumerate(branches)) {
            if (!b.isAccumulator) {
              map.map(old.getArgument(b.input), args[i]);
              continue;
            }
            Value wide = nested.create<arith::SIToFPOp>(l, f32, args[i]);
            Value scaled = nested.create<arith::MulFOp>(
                l, wide,
                nested.create<arith::ConstantOp>(l, nested.getF32FloatAttr(tScale)));
            map.map(b.scaled.getResult(), scaled);
            skip.push_back(b.scaled.getOperation());
            skip.push_back(b.toFloat.getOperation());
            if (b.withBias)
              skip.push_back(b.withBias.getOperation());
          }
          for (Operation &op : old.without_terminator()) {
            if (llvm::is_contained(skip, &op))
              continue;
            nested.clone(op, map);
          }
          auto yield = cast<linalg::YieldOp>(old.getTerminator());
          nested.create<linalg::YieldOp>(l, map.lookup(yield.getOperand(0)));
        });

    rewriter.replaceOp(generic, added.getResults());
    return success();
  }

private:
  /// The saturating narrowing back to i8, which is what the requantization the
  /// conversion matches has to end in.
  static Value narrow(OpBuilder &b, Location loc, Value scaled) {
    auto i32 = b.getI32Type();
    Value round = b.create<math::RoundEvenOp>(loc, scaled);
    Value wide = b.create<arith::FPToSIOp>(loc, i32, round);
    Value lo = b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(-128));
    Value hi = b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(127));
    Value clamped = b.create<arith::MinSIOp>(
        loc, b.create<arith::MaxSIOp>(loc, wide, lo), hi);
    return b.create<arith::TruncIOp>(loc, b.getI8Type(), clamped);
  }

  /// `trunci(clamp(fptosi(roundeven(relu?(a + b) / so))))`, where each side is
  /// an operand widened and scaled.
  static bool matchResidual(linalg::GenericOp generic,
                            SmallVectorImpl<Branch> *branches, double *outScale,
                            bool *relu) {
    Block &body = generic.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return false;
    auto trunc = yield.getOperand(0).getDefiningOp<arith::TruncIOp>();
    if (!trunc || !trunc.getType().isInteger(8))
      return false;
    auto toInt = skipClamp(trunc.getIn()).getDefiningOp<arith::FPToSIOp>();
    if (!toInt)
      return false;
    auto round = toInt.getIn().getDefiningOp<math::RoundEvenOp>();
    if (!round)
      return false;
    auto div = round.getOperand().getDefiningOp<arith::DivFOp>();
    if (!div || !isConstantFloat(div.getRhs(), outScale) || !(*outScale > 0.0))
      return false;

    Value sum = div.getLhs();
    if (auto sel = sum.getDefiningOp<arith::SelectOp>()) {
      auto cmp = sel.getCondition().getDefiningOp<arith::CmpFOp>();
      if (!cmp || !matchPattern(sel.getFalseValue(), m_AnyZeroFloat()))
        return false;
      *relu = true;
      sum = sel.getTrueValue();
    }
    auto add = sum.getDefiningOp<arith::AddFOp>();
    if (!add)
      return false;

    for (Value side : {add.getLhs(), add.getRhs()}) {
      Branch b;
      auto mul = side.getDefiningOp<arith::MulFOp>();
      if (!mul)
        return false;
      Value conv = mul.getLhs();
      if (!isConstantFloat(mul.getRhs(), &b.scale)) {
        conv = mul.getRhs();
        if (!isConstantFloat(mul.getLhs(), &b.scale))
          return false;
      }
      if (!(b.scale > 0.0))
        return false;
      b.scaled = mul;
      b.toFloat = conv.getDefiningOp<arith::SIToFPOp>();
      if (!b.toFloat || !b.toFloat->hasOneUse())
        return false;

      Value acc = b.toFloat.getIn();
      if (auto sum = acc.getDefiningOp<arith::AddIOp>()) {
        if (!sum->hasOneUse())
          return false;
        auto lhs = dyn_cast<BlockArgument>(sum.getLhs());
        auto rhs = dyn_cast<BlockArgument>(sum.getRhs());
        if (!lhs || !rhs)
          return false;
        b.withBias = sum;
        b.input = lhs.getArgNumber();
        b.bias = rhs.getArgNumber();
        acc = lhs;
      } else {
        auto arg = dyn_cast<BlockArgument>(acc);
        if (!arg)
          return false;
        b.input = arg.getArgNumber();
      }
      if (b.input >= generic.getInputs().size())
        return false;
      Type elem = getElementTypeOrSelf(generic.getInputs()[b.input].getType());
      if (elem.isInteger(32))
        b.isAccumulator = true;
      else if (!elem.isInteger(8) || b.bias >= 0)
        return false;
      branches->push_back(b);
    }
    if (branches->size() != 2 || (*branches)[0].input == (*branches)[1].input)
      return false;
    // Every operand accounted for, or something else is being computed.
    unsigned used = 0;
    for (const Branch &b : *branches)
      used += 1 + (b.bias >= 0 ? 1 : 0);
    return used == generic.getInputs().size();
  }
};

class SplitResidualAdd : public impl::SplitResidualAddBase<SplitResidualAdd> {
public:
  using impl::SplitResidualAddBase<SplitResidualAdd>::SplitResidualAddBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SplitTail>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
