//===- SelectToMinMaxPass.cpp ------------------------------------*- C++ -*-===//
//
// A relu written as a comparison and a select is a branch; `fmax.s` is not.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_SELECTTOMINMAX
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

constexpr unsigned kDepth = 8;

static bool neverNaN(Value v, unsigned depth);

/// Never a NaN *and* never an infinity. Needed wherever an infinity could meet
/// its opposite, or a zero, and make a NaN out of two operands that are not.
static bool alwaysFinite(Value v, unsigned depth);

/// The values a `linalg` body reads come from its operands. A constant global
/// is the one kind whose contents can be read here and checked.
static std::optional<bool> globalIsFinite(Value memref) {
  auto get = memref.getDefiningOp<memref::GetGlobalOp>();
  if (!get)
    return std::nullopt;
  auto module = get->getParentOfType<ModuleOp>();
  auto global = module.lookupSymbol<memref::GlobalOp>(get.getNameAttr());
  if (!global || !global.getConstant())
    return std::nullopt;
  auto dense = llvm::dyn_cast_or_null<DenseElementsAttr>(
      global.getInitialValue().value_or(Attribute()));
  if (!dense || !llvm::isa<FloatType>(getElementTypeOrSelf(dense.getType())))
    return std::nullopt;
  for (const llvm::APFloat &f : dense.getValues<llvm::APFloat>())
    if (!f.isFinite())
      return false;
  return true;
}

static std::optional<bool> blockArgumentIsFinite(Value v) {
  auto arg = llvm::dyn_cast<BlockArgument>(v);
  if (!arg)
    return std::nullopt;
  // A linalg body's arguments are its operands, inputs then inits, in order.
  auto linalgOp =
      llvm::dyn_cast_or_null<linalg::LinalgOp>(arg.getOwner()->getParentOp());
  if (!linalgOp || arg.getOwner() != linalgOp.getBlock() ||
      arg.getArgNumber() >= linalgOp->getNumOperands())
    return std::nullopt;
  return globalIsFinite(linalgOp->getOperand(arg.getArgNumber()));
}

static bool alwaysFinite(Value v, unsigned depth) {
  if (depth == 0)
    return false;
  if (std::optional<bool> known = blockArgumentIsFinite(v))
    return *known;
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  llvm::APFloat c(0.0f);
  if (matchPattern(v, m_ConstantFloat(&c)))
    return c.isFinite();
  // An integer that the float type can hold. `i32` into `f32` always can;
  // a wide enough integer into a narrow enough float rounds to an infinity.
  if (llvm::isa<arith::SIToFPOp, arith::UIToFPOp>(op)) {
    auto intType = llvm::dyn_cast<IntegerType>(
        getElementTypeOrSelf(op->getOperand(0).getType()));
    auto floatType =
        llvm::dyn_cast<FloatType>(getElementTypeOrSelf(v.getType()));
    if (!intType || !floatType)
      return false;
    llvm::APFloat largest =
        llvm::APFloat::getLargest(floatType.getFloatSemantics());
    llvm::APFloat bound(floatType.getFloatSemantics());
    bound.convertFromAPInt(llvm::APInt::getOneBitSet(intType.getWidth() + 1,
                                                     intType.getWidth()),
                           /*IsSigned=*/false, llvm::APFloat::rmTowardPositive);
    return bound.isFinite() && bound <= largest;
  }
  if (llvm::isa<arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp,
                arith::MinimumFOp>(op))
    return alwaysFinite(op->getOperand(0), depth - 1) &&
           alwaysFinite(op->getOperand(1), depth - 1);
  return false;
}

static bool neverNaN(Value v, unsigned depth) {
  if (depth == 0)
    return false;
  if (alwaysFinite(v, depth))
    return true;
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  // A product of two finite numbers can overflow to an infinity, but it is
  // never a NaN: that would take an infinity times a zero.
  if (llvm::isa<arith::MulFOp>(op))
    return alwaysFinite(op->getOperand(0), depth - 1) &&
           alwaysFinite(op->getOperand(1), depth - 1);
  // Likewise a sum, as long as only one side can be an infinity -- two
  // opposite infinities are the only way to reach a NaN from non-NaNs.
  if (llvm::isa<arith::AddFOp, arith::SubFOp>(op))
    return (alwaysFinite(op->getOperand(0), depth - 1) &&
            neverNaN(op->getOperand(1), depth - 1)) ||
           (neverNaN(op->getOperand(0), depth - 1) &&
            alwaysFinite(op->getOperand(1), depth - 1));
  // `fma` rounds once, so the product is exact going into the sum: finite
  // times finite cannot be an infinity there, and cannot meet one.
  if (llvm::isa<math::FmaOp>(op))
    return alwaysFinite(op->getOperand(0), depth - 1) &&
           alwaysFinite(op->getOperand(1), depth - 1) &&
           neverNaN(op->getOperand(2), depth - 1);
  // These hand back the operand that is not a NaN.
  if (llvm::isa<arith::MaxNumFOp, arith::MinNumFOp>(op))
    return neverNaN(op->getOperand(0), depth - 1) ||
           neverNaN(op->getOperand(1), depth - 1);
  return false;
}


/// True when `v` is only ever read by float arithmetic that ends in a
/// conversion to an integer -- the same condition
/// `--hoist-invariant-reciprocal` and `--combine-channel-affine` turn on.
static bool reachesOnlyIntegerConversion(Value v, unsigned depth = 0) {
  if (depth > 16)
    return false;
  for (Operation *user : v.getUsers()) {
    if (llvm::isa<arith::FPToSIOp, arith::FPToUIOp>(user))
      continue;
    if (llvm::isa<arith::CmpFOp>(user))
      continue; // yields an i1; it cannot carry the value on
    if (!llvm::isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                   arith::NegFOp, arith::MaximumFOp, arith::MaxNumFOp,
                   arith::MinimumFOp, arith::MinNumFOp, arith::SelectOp,
                   math::RoundEvenOp, math::FmaOp>(user))
      return false;
    if (user->getNumResults() != 1 ||
        !reachesOnlyIntegerConversion(user->getResult(0), depth + 1))
      return false;
  }
  return true;
}

/// `x > c ? x : c` becomes `arith.maxnumf x, c`, which is one `fmax.s`.
///
/// A relu reaches here as a compare and a select -- that is what torch-mlir
/// emits -- and lowers to `fle.s`, a branch, an `fmv.s` and a jump: four
/// instructions, a data-dependent branch, and three steps on the dependency
/// chain between the bias add and the store. `fmax.s` is one of each. Measured
/// on a 2048-element dequantize tail: **30.4 -> 27.5 cycles an element**, and
/// with [[FuseMultiplyAdd]] as well, 23.6.
///
/// **The NaN rule.** `maxnumf` hands back whichever operand is not a NaN; the
/// select hands back whichever side its comparison fell to, and a NaN makes an
/// ordered comparison false and an unordered one true. So the two agree exactly
/// when the operand the select would yield on a NaN is not one: the false value
/// for an ordered predicate, the true value for an unordered one. An ordered
/// relu therefore needs nothing proved. torch-mlir emits the unordered form.
///
/// That leaves proving the value is not a NaN, which in a dequantize tail it
/// cannot be: an `i32` accumulator through `sitofp` is finite, a finite scale
/// cannot make a NaN of it, and the bias is a constant global whose contents
/// this pass reads.
///
/// **The zero rule.** `fmax.s(+0, -0)` is `+0` where the select would have kept
/// the `-0` it was comparing against, so a maximum against a negative zero and
/// a minimum against a positive one are left alone. A relu's bound is `+0`.
class CompareAndSelect : public OpRewritePattern<arith::SelectOp> {
public:
  CompareAndSelect(MLIRContext *ctx, bool licenseByDestination)
      : OpRewritePattern(ctx), licenseByDestination(licenseByDestination) {}

  LogicalResult matchAndRewrite(arith::SelectOp sel,
                                PatternRewriter &rewriter) const final {
    auto cmp = sel.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!cmp || !cmp->hasOneUse())
      return failure();
    Value lhs = cmp.getLhs(), rhs = cmp.getRhs();
    Value trueValue = sel.getTrueValue(), falseValue = sel.getFalseValue();

    bool straight = trueValue == lhs && falseValue == rhs;
    bool swapped = trueValue == rhs && falseValue == lhs;
    if (!straight && !swapped)
      return failure();

    bool greater, ordered;
    switch (cmp.getPredicate()) {
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::OGE:
      greater = true, ordered = true;
      break;
    case arith::CmpFPredicate::UGT:
    case arith::CmpFPredicate::UGE:
      greater = true, ordered = false;
      break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::OLE:
      greater = false, ordered = true;
      break;
    case arith::CmpFPredicate::ULT:
    case arith::CmpFPredicate::ULE:
      greater = false, ordered = false;
      break;
    default:
      return failure();
    }
    bool isMax = greater == straight;

    // The operand a NaN would send the select to has to not be one.
    //
    // **Tried and reverted: licensing it by where the result goes.** The two
    // forms differ only on a NaN, and where the result reaches nothing but a
    // conversion to an integer that difference is unobservable, because
    // `fptosi` of a NaN is poison already -- the same argument
    // `--hoist-invariant-reciprocal` makes for its reciprocal. It lets a batch
    // norm whose input is an f32 buffer end in one `fmax.s` instead of a
    // compare, a branch, a move and a jump; DenseNet has 593 of those over 8.0
    // million elements, and all 598 compare-and-selects in the model became
    // `arith.maxnumf`.
    //
    // Correct, and **slower**, for the reason the integer clamp below is:
    // `fmax.s` sits *on* the dependency chain between the multiply and the
    // convert, while the branch it replaces hangs off it and is predicted
    // not-taken. Measured with the two builds alternated in one board session,
    // which is the only way a difference this size can be read:
    //
    //     densenet121  rsqrt-only  5073.15 ms   5071.27 ms
    //     densenet121  +this       5221.06 ms
    //
    // against a run-to-run spread of 0.2% on the same binary -- so 2.9% slower,
    // fourteen times the noise. Byte-identical to the CPU reference over forty
    // runs either way.
    //
    // **And re-measured on 2026-09-15, where it wins.** Between the two
    // measurements the loop around it lost three steps of dependency chain --
    // the accumulator came out of memory, the windows were straightened, the
    // per-channel affine became one multiply-add -- and the body went from two
    // elements to four. With four short chains interleaved there is slack to
    // hide an `fmax.s` in, and a mispredicting branch to save on every one of
    // them:
    //
    //     densenet121   853.28 ms -> 820.95 ms   (-3.8%)
    //     whole set    2845.76 ms -> 2813.30 ms  (-1.1%)
    //
    // Every other model within +-0.3%, all byte-identical. **The number that was
    // right about one loop shape was wrong about the next one**; a constant like
    // this is a fact about the body, not about the pass.
    //
    // **Re-measured 2026-09-15, and it wins now.** The number above was taken
    // when the loop around it was much longer: the reduction accumulator still
    // went through memory, the windows were still loops, the per-channel affine
    // was still three operations an element, and the body held **two**
    // elements. On the chains those left -- `fmadd`, the activation, the
    // convert -- and with **four** of them interleaved, there is slack to hide
    // an `fmax.s` in and a mispredicting branch to save on every one. The
    // licence is the one `--saturate-constant-casts` uses: where the value
    // reaches nothing but a conversion to an integer, a NaN is already poison,
    // so handing back the other operand is a refinement.
    if (!neverNaN(ordered ? falseValue : trueValue, kDepth) &&
        !(licenseByDestination && reachesOnlyIntegerConversion(sel.getResult())))
      return failure();

    // A zero whose sign min/max would not have picked.
    for (Value v : {lhs, rhs}) {
      llvm::APFloat c(0.0f);
      if (matchPattern(v, m_ConstantFloat(&c)) && c.isZero() &&
          c.isNegative() == isMax)
        return failure();
    }

    if (isMax)
      rewriter.replaceOpWithNewOp<arith::MaxNumFOp>(sel, lhs, rhs);
    else
      rewriter.replaceOpWithNewOp<arith::MinNumFOp>(sel, lhs, rhs);
    return success();
  }

private:
  bool licenseByDestination;
};

/// **Tried twice and reverted twice: the integer clamp moved into the float.**
///
/// A quantization tail ends `fptosi`, `maxsi(_, -128)`, `minsi(_, 127)`. This
/// board's Rocket is plain `rv64gc` -- no `Zbb`, so no `max`/`min` instruction
/// and no conditional move -- and each clamp is a compare, a branch and an
/// `li`. `fmax.s`/`fmin.s` are one instruction each and never branch, and
/// clamping in the float is the **same integer**: clamping commutes with
/// rounding when the bounds are integers. `scripts/clamp_check.c` verifies that
/// over every f32 bit pattern between -300 and 300 -- 2,267,807,744 values, no
/// mismatch -- and the accelerator IR is byte-identical either way.
///
/// It is exact and it is **slower**, because on an in-order single-issue core
/// an element costs its *dependency chain*, not its instruction count.
/// `fmax.s`/`fmin.s` sit between the multiply and `fcvt.w.s` and lengthen it;
/// the two branches they replace hang off the chain entirely and are predicted
/// not-taken. Measured on the U280, every model 0 of 40 against the CPU
/// reference:
///
/// | | before | after |
/// |---|---|---|
/// | `gmin` | 10.64 ms | 11.53 |
/// | `gmid` | 11.46 | 12.13 |
/// | `resnet18` | 68.49 | 73.06 |
/// | `mnasnet0_5` | 74.10 | 75.49 |
/// | `squeezenet1_1` | 47.77 | 48.65 |
/// | `lstm` | 18.29 | 18.55 |
///
/// An earlier session measured the same rewrite across 26 models at
/// 225.8 -> 231.1 ms, 22 of them slower. **This is the second time it has been
/// written and reverted**; see `gemmlir-the-chain-not-the-count` in the
/// project's notes, and read that before trading instructions again.
///
/// Worth revisiting only on a core with `Zbb`, where the integer clamp is two
/// instructions and no branches.

class SelectToMinMax : public impl::SelectToMinMaxBase<SelectToMinMax> {
public:
  using impl::SelectToMinMaxBase<SelectToMinMax>::SelectToMinMaxBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, math::MathDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<CompareAndSelect>(&getContext(), licenseByDestination);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
