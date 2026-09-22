//===- RequantizeBeforePoolingPass.cpp ---------------------*- C++ -*-===//
//
// Moves a monotonic elementwise operation ahead of a max-pool.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_REQUANTIZEBEFOREPOOLING
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

static bool isConst(Value v) {
  Attribute a;
  return matchPattern(v, m_Constant(&a));
}

static bool isPositiveConst(Value v) {
  llvm::APFloat f(0.0f);
  if (matchPattern(v, m_ConstantFloat(&f)))
    return f.isFiniteNonZero() && !f.isNegative();
  llvm::APInt i;
  if (matchPattern(v, m_ConstantInt(&i)))
    return i.isStrictlyPositive();
  return false;
}

/// A `trunci` is only non-decreasing where it cannot wrap, which is what a
/// clamp to the destination's own range guarantees.
static bool clampedToDestination(arith::TruncIOp trunc) {
  unsigned bits = llvm::cast<IntegerType>(
                      getElementTypeOrSelf(trunc.getType())).getWidth();
  auto isBound = [&](Value v, int64_t want) {
    llvm::APInt i;
    return matchPattern(v, m_ConstantInt(&i)) && i.getSExtValue() == want;
  };
  auto hi = trunc.getIn().getDefiningOp<arith::MinSIOp>();
  if (!hi || !isBound(hi.getRhs(), llvm::maxIntN(bits)))
    return false;
  auto lo = hi.getLhs().getDefiningOp<arith::MaxSIOp>();
  return lo && isBound(lo.getRhs(), llvm::minIntN(bits));
}

/// True when `v` is a non-decreasing function of `arg`.
static bool isNonDecreasing(Value v, BlockArgument arg, unsigned depth = 0) {
  if (v == arg)
    return true;
  if (depth > 32)
    return false;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  auto through = [&](Value x) { return isNonDecreasing(x, arg, depth + 1); };

  // Shape- and sign-preserving conversions.
  if (llvm::isa<math::RoundEvenOp, arith::FPToSIOp, arith::SIToFPOp,
                arith::ExtSIOp, arith::ExtFOp>(def))
    return through(def->getOperand(0));
  if (auto trunc = llvm::dyn_cast<arith::TruncIOp>(def))
    return clampedToDestination(trunc) && through(trunc.getIn());

  // A constant on one side, the value on the other.
  auto binary = [&](Value lhs, Value rhs, bool rhsMustBePositive,
                    bool lhsMayBeConst) {
    if (isConst(rhs) && (!rhsMustBePositive || isPositiveConst(rhs)))
      return through(lhs);
    if (lhsMayBeConst && isConst(lhs) && !rhsMustBePositive)
      return through(rhs);
    return false;
  };
  if (auto op = llvm::dyn_cast<arith::MulFOp>(def))
    return (isPositiveConst(op.getRhs()) && through(op.getLhs())) ||
           (isPositiveConst(op.getLhs()) && through(op.getRhs()));
  if (auto op = llvm::dyn_cast<arith::DivFOp>(def))
    return isPositiveConst(op.getRhs()) && through(op.getLhs());
  if (llvm::isa<arith::AddFOp, arith::AddIOp, arith::MaximumFOp,
                arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp,
                arith::MaxSIOp, arith::MinSIOp>(def))
    return binary(def->getOperand(0), def->getOperand(1), false, true);
  if (auto op = llvm::dyn_cast<arith::SubFOp>(def))
    return isConst(op.getRhs()) && through(op.getLhs());
  return false;
}

/// `quantize(maxpool(x))` is `maxpool(quantize(x))`.
///
/// A quantized network as a frontend writes it pools in f32 and requantizes
/// afterwards, which leaves the convolution's tail -- dequantize, bias,
/// activation -- with no i8 result to fold into, so the whole layer stays in
/// software. `max` commutes with any non-decreasing function, so the
/// requantization can go first; the pool then runs on i8 as well, a quarter of
/// the memory traffic.
///
/// The pool's identity changes with the element type. Only an integer result is
/// handled, where the type's own minimum is below every value the body can
/// produce -- and that is exactly the value a max-pool's accumulator starts at.
class RequantizeBeforeMaxPool : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    // The frontend flattens the pooled activation before the classifier, and
    // the layout rewrite leaves a transpose in front of that, so what
    // requantizes the pool reads it through a chain of shape-only operations.
    // Those are replayed after the moved pool rather than matched away.
    Value src = generic.getInputs()[0];
    SmallVector<Operation *> chain;
    while (Operation *def = src.getDefiningOp()) {
      if (!def->hasOneUse() ||
          !llvm::isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp,
                     linalg::TransposeOp>(def))
        break;
      chain.push_back(def);
      src = def->getOperand(0);
    }
    // Either spelling of a max-pool. `max` commutes with a non-decreasing
    // function whatever order the axes are written in, and the rewrite below
    // rebuilds whichever one it found -- so nothing here is about layout. An
    // NCHW pool is what a model whose convolution became an im2col matmul is
    // left with: `--conv-nchw-to-nhwc` is not in the quantized pipeline, so
    // nothing else turns it over.
    Operation *poolOp = src.getDefiningOp();
    if (!poolOp || !llvm::isa<linalg::PoolingNhwcMaxOp,
                              linalg::PoolingNchwMaxOp>(poolOp))
      return failure();
    // GoogLeNet's inception pool feeds two things: the requantization, and --
    // through a pad -- **the next module's pool**. `hasOneUse` refused both of
    // the two pools that matter, which are exactly the ones whose input is a
    // concatenation of four dequantized branches.
    //
    // A second reader that is itself a max-pool is no obstacle: it will be
    // moved the same way when its own requantization is matched, and the two
    // agree on a scale because `--share-branch-quantization` already made them.
    // What it needs is the inverse on the way out, so the f32 reader still sees
    // f32 -- and `quantize(dequantize(q))` is `q` exactly for a byte, so
    // nothing is lost when that reader quantizes again.
    SmallVector<OpOperand *> otherUses;
    for (OpOperand &use : poolOp->getResult(0).getUses()) {
      Operation *user = use.getOwner();
      if (user == generic || llvm::is_contained(chain, user))
        continue;
      Operation *reader = user;
      if (auto pad = llvm::dyn_cast<tensor::PadOp>(user)) {
        if (!pad->hasOneUse())
          return failure();
        reader = *pad->getUsers().begin();
      }
      // ...or the very same requantization, which is what the second reader
      // has already become once the pool below it was moved: the inverse this
      // hands it is cancelled exactly, because `quantize(dequantize(q))` is `q`
      // for a byte.
      bool sameQuantize = false;
      if (auto other = llvm::dyn_cast<linalg::GenericOp>(reader))
        sameQuantize =
            other != generic && other.getInputs().size() == 1 &&
            other.getOutputs().size() == 1 &&
            OperationEquivalence::isRegionEquivalentTo(
                &other.getRegion(), &generic.getRegion(),
                OperationEquivalence::IgnoreLocations);
      if (!llvm::isa<linalg::PoolingNhwcMaxOp, linalg::PoolingNchwMaxOp>(reader) &&
          !sameQuantize)
        return failure();
      otherUses.push_back(&use);
    }
    auto pool = llvm::cast<linalg::LinalgOp>(poolOp);
    if (pool.getDpsInputs().size() != 2)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    if (!llvm::all_of(generic.getIndexingMapsArray(),
                      [](AffineMap m) { return m.isIdentity(); }))
      return failure();

    auto resTy = llvm::dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    auto srcTy = llvm::dyn_cast<RankedTensorType>(pool.getDpsInputs()[0].getType());
    auto poolTy = llvm::dyn_cast<RankedTensorType>(pool->getResult(0).getType());
    if (!resTy || !srcTy || !poolTy || !resTy.hasStaticShape() ||
        !srcTy.hasStaticShape() || !poolTy.hasStaticShape())
      return failure();
    auto intTy = llvm::dyn_cast<IntegerType>(resTy.getElementType());
    if (!intTy || intTy.getWidth() > 64)
      return failure();

    Block &body = generic.getRegion().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        !body.getArguments().back().use_empty())
      return failure();
    if (!isNonDecreasing(yield.getOperand(0), body.getArgument(0)))
      return failure();

    Location loc = generic.getLoc();
    auto earlyTy = RankedTensorType::get(srcTy.getShape(), intTy);
    Value earlyInit = rewriter.create<tensor::EmptyOp>(loc, earlyTy.getShape(),
                                                       intTy);
    // The operation being moved is elementwise, so it takes the rank of
    // whatever it now reads -- which is the pool's input, not the flattened
    // view it used to see.
    unsigned rank = srcTy.getRank();
    SmallVector<AffineMap> maps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto early = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{earlyTy}, ValueRange{pool.getDpsInputs()[0]},
        ValueRange{earlyInit}, maps, iters);
    rewriter.cloneRegionBefore(generic.getRegion(), early.getRegion(),
                               early.getRegion().begin());

    Value least = rewriter.create<arith::ConstantOp>(
        loc, intTy,
        rewriter.getIntegerAttr(intTy, llvm::APInt::getSignedMinValue(
                                           intTy.getWidth())));
    auto newPoolTy = RankedTensorType::get(poolTy.getShape(), intTy);
    Value poolInit = rewriter.create<tensor::EmptyOp>(loc, newPoolTy.getShape(),
                                                      intTy);
    Value filled = rewriter.create<linalg::FillOp>(loc, ValueRange{least},
                                                   ValueRange{poolInit})
                       .getResult(0);
    // A `ValueRange` built from a brace list points into a temporary that dies
    // with the statement; this has to own its operands.
    SmallVector<Value> ins{early.getResult(0), pool.getDpsInputs()[1]};
    auto strides = poolOp->getAttrOfType<DenseIntElementsAttr>("strides");
    auto dilations = poolOp->getAttrOfType<DenseIntElementsAttr>("dilations");
    if (!strides || !dilations)
      return failure();
    Value pooled =
        llvm::isa<linalg::PoolingNhwcMaxOp>(poolOp)
            ? rewriter
                  .create<linalg::PoolingNhwcMaxOp>(loc, TypeRange{newPoolTy},
                                                    ins, ValueRange{filled},
                                                    strides, dilations)
                  ->getResult(0)
            : rewriter
                  .create<linalg::PoolingNchwMaxOp>(loc, TypeRange{newPoolTy},
                                                    ins, ValueRange{filled},
                                                    strides, dilations)
                  ->getResult(0);
    for (Operation *op : llvm::reverse(chain)) {
      auto ty = llvm::cast<RankedTensorType>(op->getResult(0).getType())
                    .clone(intTy);
      if (auto c = llvm::dyn_cast<tensor::CollapseShapeOp>(op))
        pooled = rewriter.create<tensor::CollapseShapeOp>(
            loc, ty, pooled, c.getReassociationIndices());
      else if (auto e = llvm::dyn_cast<tensor::ExpandShapeOp>(op))
        pooled = rewriter.create<tensor::ExpandShapeOp>(
            loc, ty, pooled, e.getReassociationIndices());
      else {
        auto t = llvm::cast<linalg::TransposeOp>(op);
        Value init = rewriter.create<tensor::EmptyOp>(loc, ty.getShape(), intTy);
        pooled = rewriter.create<linalg::TransposeOp>(loc, pooled, init,
                                                      t.getPermutation())
                     ->getResult(0);
      }
    }
    // Whatever else read the f32 pool gets the inverse of the quantization that
    // moved: one multiply by the scale it divided by. Only that exact body has
    // an inverse this pass can write, which is why a multi-reader pool is
    // confined to it.
    if (!otherUses.empty()) {
      auto divf = yield.getOperand(0)
                      .getDefiningOp()
                      ->getBlock()
                      ->getParentOp();
      (void)divf;
      arith::DivFOp scaleOp;
      body.walk([&](arith::DivFOp d) {
        if (d.getLhs() == body.getArgument(0))
          scaleOp = d;
      });
      FloatAttr scaleAttr;
      if (!scaleOp || !matchPattern(scaleOp.getRhs(), m_Constant(&scaleAttr)) ||
          !scaleAttr.getValue().isNormal() || scaleAttr.getValue().isNegative())
        return failure();
      Type f32 = scaleOp.getType();
      Value backInit = rewriter.create<tensor::EmptyOp>(
          loc, poolTy.getShape(), f32);
      SmallVector<AffineMap> bmaps(
          2, AffineMap::getMultiDimIdentityMap(poolTy.getRank(),
                                               rewriter.getContext()));
      SmallVector<utils::IteratorType> biters(poolTy.getRank(),
                                              utils::IteratorType::parallel);
      auto back = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{poolTy}, ValueRange{pooled}, ValueRange{backInit},
          bmaps, biters,
          [&](OpBuilder &b, Location nested, ValueRange args) {
            Value f = b.create<arith::SIToFPOp>(nested, f32, args[0]);
            Value s = b.create<arith::ConstantOp>(nested, scaleAttr);
            b.create<linalg::YieldOp>(
                nested, ValueRange{b.create<arith::MulFOp>(nested, f, s)});
          });
      for (OpOperand *use : otherUses)
        rewriter.modifyOpInPlace(use->getOwner(),
                                 [&] { use->set(back.getResult(0)); });
    }

    rewriter.replaceOp(generic, pooled);
    return success();
  }
};

class RequantizeBeforePooling
    : public impl::RequantizeBeforePoolingBase<RequantizeBeforePooling> {
public:
  using impl::RequantizeBeforePoolingBase<
      RequantizeBeforePooling>::RequantizeBeforePoolingBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RequantizeBeforeMaxPool>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
