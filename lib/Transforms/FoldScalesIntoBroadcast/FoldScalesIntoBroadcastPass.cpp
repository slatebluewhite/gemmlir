//===- FoldScalesIntoBroadcastPass.cpp ---------------------------*- C++ -*-===//
//
// Scale the operand there are fewer of.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FOLDSCALESINTOBROADCAST
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

static bool floatConst(Value v, APFloat &out) {
  FloatAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return false;
  out = attr.getValue();
  return true;
}

/// One step of a pure scaling -- `x * c`, `x / c` or `-x`. Anything with an
/// offset stops the walk, because an offset does not commute with moving the
/// scale onto a different operand.
/// One step of the chain *above* the multiply, as an affine map.
///
/// The invariant is `top == factor * (root * buffer) + offset`. A pure scaling
/// moves both; a constant offset moves only itself -- and it can move at all,
/// which is what the scaling-only walk below could not do: `(x*r + b)/s` is
/// `x*(r/s) + b/s` exactly when `b` is a number the compiler has.
static bool affineStep(Operation *op, Value cur, Value &var, APFloat &factor,
                       APFloat &offset) {
  const llvm::fltSemantics &sem = factor.getSemantics();
  auto rm = APFloat::rmNearestTiesToEven;
  auto cst = [&](unsigned i, APFloat &out) {
    if (!floatConst(op->getOperand(i), out))
      return false;
    bool lost;
    return out.convert(sem, rm, &lost) == APFloat::opOK;
  };
  APFloat k(sem);
  if (auto neg = llvm::dyn_cast<arith::NegFOp>(op)) {
    if (neg.getOperand() != cur)
      return false;
    var = neg.getResult();
    factor.changeSign();
    offset.changeSign();
    return true;
  }
  if (llvm::isa<arith::MulFOp>(op)) {
    unsigned other;
    if (op->getOperand(0) == cur && cst(1, k))
      other = 1;
    else if (op->getOperand(1) == cur && cst(0, k))
      other = 0;
    else
      return false;
    (void)other;
    factor.multiply(k, rm);
    offset.multiply(k, rm);
    var = op->getResult(0);
    return true;
  }
  if (llvm::isa<arith::DivFOp>(op)) {
    // Only division *by* the constant: the other way round is not affine.
    if (op->getOperand(0) != cur || !cst(1, k) || k.isZero())
      return false;
    factor.divide(k, rm);
    offset.divide(k, rm);
    var = op->getResult(0);
    return true;
  }
  if (llvm::isa<arith::AddFOp>(op)) {
    if (op->getOperand(0) == cur && cst(1, k))
      ;
    else if (op->getOperand(1) == cur && cst(0, k))
      ;
    else
      return false;
    offset.add(k, rm);
    var = op->getResult(0);
    return true;
  }
  if (llvm::isa<arith::SubFOp>(op)) {
    // `cur - c` moves; `c - cur` would flip the sign of the product too, and
    // there is no shape in the set that asks for it.
    if (op->getOperand(0) != cur || !cst(1, k))
      return false;
    offset.subtract(k, rm);
    var = op->getResult(0);
    return true;
  }
  return false;
}

static bool scaleStep(Operation *op, Value &var, APFloat &factor) {
  const llvm::fltSemantics &sem = factor.getSemantics();
  auto rm = APFloat::rmNearestTiesToEven;
  auto cst = [&](unsigned i, APFloat &out) {
    if (!floatConst(op->getOperand(i), out))
      return false;
    bool lost;
    return out.convert(sem, rm, &lost) == APFloat::opOK;
  };
  APFloat k(sem);
  if (auto neg = llvm::dyn_cast<arith::NegFOp>(op)) {
    var = neg.getOperand();
    factor.changeSign();
    return true;
  }
  if (llvm::isa<arith::MulFOp>(op)) {
    if (cst(1, k)) {
      var = op->getOperand(0);
    } else if (cst(0, k)) {
      var = op->getOperand(1);
    } else {
      return false;
    }
    factor.multiply(k, rm);
    return true;
  }
  if (llvm::isa<arith::DivFOp>(op)) {
    if (!cst(1, k))
      return false;
    var = op->getOperand(0);
    factor.divide(k, rm);
    return true;
  }
  return false;
}

/// The licence: the value reaches nothing but a conversion to an integer,
/// through monotone operations only, so a few ulps can move nothing but a
/// rounding boundary. The same one the reciprocal takes.
static bool reachesIntegerConversion(Value v) {
  SmallVector<Value> work{v};
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    if (cur.use_empty())
      return false;
    for (Operation *user : cur.getUsers()) {
      if (llvm::isa<arith::FPToSIOp, arith::FPToUIOp>(user))
        continue;
      if (llvm::isa<math::RoundEvenOp, math::RoundOp, arith::MaxNumFOp,
                    arith::MinNumFOp, arith::MaximumFOp, arith::MinimumFOp>(
              user)) {
        work.push_back(user->getResult(0));
        continue;
      }
      return false;
    }
  }
  return true;
}

static int64_t elementsOf(Value v) {
  auto ty = llvm::dyn_cast<MemRefType>(v.getType());
  return ty && ty.hasStaticShape() ? ty.getNumElements() : -1;
}

/// EfficientNet's squeeze-and-excitation is
///
/// ```
///   (x * input_scale) * gate[c] / output_scale
/// ```
///
/// -- three floating-point multiplies an element, two of them by numbers the
/// compiler knows. `gate` is **96 numbers** against 16x16x96 elements, so the
/// two constants belong in it: one small loop, and the element loop drops from
/// four operations to two. It is 26% of that model's elementwise work.
///
/// `--combine-constant-scales` cannot reach these because the running
/// coefficient sits *between* the two constants and breaks the chain. The rule
/// here is the same one [[gemmlir-two-numbers-per-channel]] used: when a loop
/// multiplies by something it reads through a map that drops loop dimensions,
/// the constants around it belong on that side of the multiply.
///
/// Only pure scaling moves -- an offset does not commute with the multiply --
/// and the licence is the usual one, the value reaching nothing but a
/// conversion to an integer.
class FoldScalesIntoBroadcast : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic->getNumResults() != 0 || generic.getOutputs().size() != 1)
      return failure();
    int64_t outElements = elementsOf(generic.getOutputs()[0]);
    if (outElements <= 0)
      return failure();

    Block &body = generic.getRegion().front();
    unsigned numIn = generic.getInputs().size();

    for (Operation &op : body.without_terminator()) {
      auto mul = llvm::dyn_cast<arith::MulFOp>(&op);
      if (!mul)
        continue;
      Type ty = mul.getType();
      if (!ty.isF32() && !ty.isF64())
        continue;
      const llvm::fltSemantics &sem =
          ty.isF64() ? APFloat::IEEEdouble() : APFloat::IEEEsingle();

      // One operand has to be a value read from a buffer there are fewer of.
      for (unsigned side = 0; side < 2; side++) {
        auto arg = llvm::dyn_cast<BlockArgument>(mul.getOperand(side));
        if (!arg || arg.getOwner() != &body || arg.getArgNumber() >= numIn)
          continue;
        Value operand = generic.getInputs()[arg.getArgNumber()];
        int64_t n = elementsOf(operand);
        if (n <= 0 || n >= outElements)
          continue;
        auto opTy = llvm::cast<MemRefType>(operand.getType());
        if (!opTy.getLayout().isIdentity() || opTy.getElementType() != ty)
          continue;

        // Walk the pure scaling below the multiply and above it.
        APFloat k(sem, 1);
        Value root = mul.getOperand(1 - side);
        while (Operation *def = root.getDefiningOp()) {
          Value next;
          if (!root.hasOneUse() || !scaleStep(def, next, k))
            break;
          root = next;
        }
        Value top = mul.getResult();
        APFloat off(sem, 0);
        while (top.hasOneUse()) {
          Operation *user = *top.getUsers().begin();
          Value next;
          APFloat probeK = k, probeOff = off;
          if (!affineStep(user, top, next, probeK, probeOff))
            break;
          k = probeK;
          off = probeOff;
          top = next;
        }
        if (k.compare(APFloat(sem, 1)) == APFloat::cmpEqual)
          continue;
        if (!k.isFinite() || k.isZero() || !off.isFinite())
          continue;
        if (!reachesIntegerConversion(top))
          continue;

        // One small loop scales the buffer; the element loop keeps one multiply.
        Location loc = generic.getLoc();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(generic);
        Value scaled = rewriter.create<memref::AllocOp>(
            loc, opTy, rewriter.getI64IntegerAttr(64));
        unsigned rank = opTy.getRank();
        SmallVector<AffineMap> maps(
            2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
        SmallVector<utils::IteratorType> iters(rank,
                                               utils::IteratorType::parallel);
        APFloat kk = k;
        rewriter.create<linalg::GenericOp>(
            loc, TypeRange{}, ValueRange{operand}, ValueRange{scaled}, maps,
            iters, [&](OpBuilder &b, Location nested, ValueRange args) {
              Value c = b.create<arith::ConstantOp>(
                  nested, b.getFloatAttr(ty, kk));
              b.create<linalg::YieldOp>(
                  nested,
                  ValueRange{b.create<arith::MulFOp>(nested, args[0], c)});
            });

        rewriter.setInsertionPoint(mul);
        Value replacement =
            rewriter.create<arith::MulFOp>(loc, root, mul.getOperand(side));
        if (!off.isZero()) {
          APFloat offCopy = off;
          Value c = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getFloatAttr(ty, offCopy));
          replacement = rewriter.create<arith::AddFOp>(loc, replacement, c);
        }
        rewriter.replaceAllUsesWith(top, replacement);
        rewriter.modifyOpInPlace(
            generic, [&] { generic->setOperand(arg.getArgNumber(), scaled); });
        return success();
      }
    }
    return failure();
  }
};

class FoldScalesIntoBroadcast_Pass
    : public impl::FoldScalesIntoBroadcastBase<FoldScalesIntoBroadcast_Pass> {
public:
  using impl::FoldScalesIntoBroadcastBase<
      FoldScalesIntoBroadcast_Pass>::FoldScalesIntoBroadcastBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, arith::ArithDialect,
                    math::MathDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldScalesIntoBroadcast>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
