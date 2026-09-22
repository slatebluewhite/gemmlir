//===- BatchNormInFixedPointPass.cpp -----------------------------*- C++ -*-===//
//
// A batch norm on a dequantized byte is integer arithmetic.
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

#include <cmath>

namespace mlir::gemmlir {

#define GEN_PASS_DEF_BATCHNORMINFIXEDPOINT
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// What the tail does to one value, in the arithmetic the board would use.
struct Tail {
  bool relu;
  int32_t lo, hi;
};

static float roundEven(float v) { return std::nearbyint(v); }

static int32_t floatAnswer(int8_t q, float s, float a, float b, const Tail &t) {
  float x = (float)q * s;       // the dequantize: one rounding
  float v = x * a;              // `mulf`
  v = v + b;                    // `addf`
  if (t.relu)
    v = v > 0.0f ? v : 0.0f;
  int32_t i = (int32_t)roundEven(v);
  if (i < t.lo)
    i = t.lo;
  if (i > t.hi)
    i = t.hi;
  return i;
}

static int32_t fixedAnswer(int8_t q, int32_t m, int32_t n, int shift,
                           const Tail &t) {
  int64_t v = ((int64_t)m * (int64_t)q + (int64_t)n) >> shift;
  int32_t i = (int32_t)v;
  if (t.relu && i < 0)
    i = 0;
  if (i < t.lo)
    i = t.lo;
  if (i > t.hi)
    i = t.hi;
  return i;
}

static DenseElementsAttr constantOf(Value v, ModuleOp module) {
  auto get = v.getDefiningOp<memref::GetGlobalOp>();
  if (!get)
    return nullptr;
  auto global = module.lookupSymbol<memref::GlobalOp>(get.getNameAttr());
  if (!global || !global.getConstant())
    return nullptr;
  return llvm::dyn_cast_or_null<DenseElementsAttr>(global.getInitialValueAttr());
}

/// DenseNet's batch norm is 72% of the model and 27 cycles an element for an
/// eight-instruction body. Shortening it was measured five ways and every one
/// bought 1-2%; what it is actually paying for is the **floating point**.
///
/// `--fold-constant-elementwise` made the per-channel coefficients compile-time
/// constants, and the value they multiply is a dequantized byte -- so the whole
/// tail is a function of one byte and one channel, and fixed point can do it
/// with integers only:
///
/// ```
///   lb / mul / add / srai / clamp / sb      instead of
///   flw flw flw / fmadd / fmax / fcvt.w.s / clamp / sb
/// ```
///
/// Measured as a kernel on the board at dense block 3's shape: **-16.6%**.
///
/// **And it is exact, not approximately exact.** Both coefficients and the
/// dequantize scale are known when the model is compiled, and the input is one
/// of 256 bytes, so the pass *evaluates both forms for every channel and every
/// byte* and only rewrites when they agree everywhere. Where the shift cannot
/// be made precise enough, the site keeps its floating point.
class BatchNormInFixedPoint : public OpRewritePattern<linalg::GenericOp> {
public:
  BatchNormInFixedPoint(MLIRContext *ctx, unsigned *counter)
      : OpRewritePattern(ctx), counter(counter) {}

  LogicalResult matchAndRewrite(linalg::GenericOp bn,
                                PatternRewriter &rewriter) const final {
    if (bn->getNumResults() != 0 || bn.getInputs().size() != 3 ||
        bn.getOutputs().size() != 1)
      return failure();
    auto module = bn->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    for (utils::IteratorType it : bn.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();

    auto outTy = llvm::dyn_cast<MemRefType>(bn.getOutputs()[0].getType());
    auto inTy = llvm::dyn_cast<MemRefType>(bn.getInputs()[0].getType());
    if (!outTy || !inTy || !outTy.getElementType().isInteger(8) ||
        !inTy.getElementType().isF32() || !inTy.hasStaticShape())
      return failure();

    // The body: mulf by one per-channel number, addf another, an optional relu,
    // then round, convert and clamp.
    Block &body = bn.getRegion().front();
    if (body.getNumArguments() != 4)
      return failure();
    BlockArgument x = body.getArgument(0);
    Tail tail{false, -128, 127};

    auto mul = llvm::dyn_cast_or_null<arith::MulFOp>(
        x.hasOneUse() ? *x.getUsers().begin() : nullptr);
    if (!mul)
      return failure();
    BlockArgument aArg = llvm::dyn_cast<BlockArgument>(
        mul.getLhs() == x ? mul.getRhs() : mul.getLhs());
    if (!aArg || aArg.getOwner() != &body || !mul.getResult().hasOneUse())
      return failure();
    auto add = llvm::dyn_cast<arith::AddFOp>(*mul.getResult().getUsers().begin());
    if (!add)
      return failure();
    BlockArgument bArg = llvm::dyn_cast<BlockArgument>(
        add.getLhs() == mul.getResult() ? add.getRhs() : add.getLhs());
    if (!bArg || bArg.getOwner() != &body)
      return failure();

    // A relu written as `cmpf` + `select` reads its input **twice**, so
    // `hasOneUse` on the value entering it is the wrong test -- it refused all
    // 593 of DenseNet's the first time [[gemmlir-two-numbers-per-channel]] met
    // this shape.
    Value cur = add.getResult();
    Operation *step = nullptr;
    for (Operation *user : cur.getUsers())
      if (llvm::isa<arith::CmpFOp>(user))
        step = user;
    if (!step) {
      if (!cur.hasOneUse())
        return failure();
      step = *cur.getUsers().begin();
    } else if (std::distance(cur.getUsers().begin(), cur.getUsers().end()) != 2) {
      return failure();
    }
    if (auto cmp = llvm::dyn_cast<arith::CmpFOp>(step)) {
      // `cmpf ugt, v, 0` then `select` -- the relu before `--select-to-minmax`.
      APFloat zero(0.0f);
      if (cmp.getPredicate() != arith::CmpFPredicate::UGT ||
          cmp.getLhs() != cur || !matchPattern(cmp.getRhs(), m_ConstantFloat(&zero)) ||
          !zero.isZero() || !cmp.getResult().hasOneUse())
        return failure();
      auto sel = llvm::dyn_cast<arith::SelectOp>(*cmp.getResult().getUsers().begin());
      if (!sel || sel.getTrueValue() != cur || !sel.getResult().hasOneUse())
        return failure();
      tail.relu = true;
      cur = sel.getResult();
      step = *cur.getUsers().begin();
    } else if (auto mx = llvm::dyn_cast<arith::MaxNumFOp>(step)) {
      APFloat zero(0.0f);
      if (!matchPattern(mx.getRhs(), m_ConstantFloat(&zero)) || !zero.isZero() ||
          !mx.getResult().hasOneUse())
        return failure();
      tail.relu = true;
      cur = mx.getResult();
      step = *cur.getUsers().begin();
    }
    auto round = llvm::dyn_cast<math::RoundEvenOp>(step);
    if (!round || !round.getResult().hasOneUse())
      return failure();
    auto toInt = llvm::dyn_cast<arith::FPToSIOp>(*round.getResult().getUsers().begin());
    if (!toInt || !toInt.getType().isInteger(32) || !toInt.getResult().hasOneUse())
      return failure();
    Value ival = toInt.getResult();
    if (auto lo = llvm::dyn_cast<arith::MaxSIOp>(*ival.getUsers().begin())) {
      IntegerAttr c;
      if (!matchPattern(lo.getRhs(), m_Constant(&c)) || !lo.getResult().hasOneUse())
        return failure();
      tail.lo = (int32_t)c.getInt();
      ival = lo.getResult();
    }
    auto hi = llvm::dyn_cast<arith::MinSIOp>(*ival.getUsers().begin());
    if (!hi) {
      // `--drop-clamp-below-relu` may already have taken the lower half; the
      // upper one is always there.
      return failure();
    }
    IntegerAttr hic;
    if (!matchPattern(hi.getRhs(), m_Constant(&hic)) || !hi.getResult().hasOneUse())
      return failure();
    tail.hi = (int32_t)hic.getInt();
    auto trunc = llvm::dyn_cast<arith::TruncIOp>(*hi.getResult().getUsers().begin());
    if (!trunc || !trunc.getType().isInteger(8))
      return failure();
    auto yield = llvm::cast<linalg::YieldOp>(body.getTerminator());
    if (yield.getNumOperands() != 1 || yield.getOperand(0) != trunc.getResult())
      return failure();

    // The coefficients have to be constants...
    DenseElementsAttr aAttr = constantOf(bn.getInputs()[aArg.getArgNumber()], module);
    DenseElementsAttr bAttr = constantOf(bn.getInputs()[bArg.getArgNumber()], module);
    if (!aAttr || !bAttr || aAttr.getNumElements() != bAttr.getNumElements())
      return failure();
    int64_t channels = aAttr.getNumElements();

    // ...and the value they multiply has to be a dequantized byte.
    Value f32buf = bn.getInputs()[0];
    linalg::GenericOp deq;
    for (Operation *user : f32buf.getUsers()) {
      auto g = llvm::dyn_cast<linalg::GenericOp>(user);
      if (!g || g == bn)
        continue;
      if (llvm::is_contained(g.getOutputs(), f32buf))
        deq = g;
    }
    float scale = 0.0f;
    Value i8src;
    if (!dequantizeOf(deq, f32buf, i8src, scale))
      return failure();
    auto srcTy = llvm::dyn_cast<MemRefType>(i8src.getType());
    if (!srcTy || !srcTy.getElementType().isInteger(8) ||
        srcTy.getShape() != inTy.getShape())
      return failure();

    // Pick the largest shift that cannot overflow, then *prove* the two forms
    // agree on every channel and every byte.
    SmallVector<float> A, B;
    for (APFloat f : aAttr.getValues<APFloat>())
      A.push_back(f.convertToFloat());
    for (APFloat f : bAttr.getValues<APFloat>())
      B.push_back(f.convertToFloat());

    SmallVector<int32_t> M(channels), N(channels);
    int shift = -1;
    for (int k = 24; k >= 8 && shift < 0; k--) {
      double limit = (double)(1u << 31);
      bool fits = true;
      for (int64_t c = 0; c < channels && fits; c++) {
        double m = std::nearbyint((double)A[c] * (double)scale * std::ldexp(1.0, k));
        double n = std::nearbyint(((double)B[c] + 0.5) * std::ldexp(1.0, k));
        if (std::abs(m) * 128.0 + std::abs(n) >= limit || std::abs(m) >= limit ||
            std::abs(n) >= limit)
          fits = false;
        else {
          M[c] = (int32_t)m;
          N[c] = (int32_t)n;
        }
      }
      if (!fits)
        continue;
      bool same = true;
      for (int64_t c = 0; c < channels && same; c++)
        for (int v = -128; v <= 127 && same; v++)
          same = floatAnswer((int8_t)v, scale, A[c], B[c], tail) ==
                 fixedAnswer((int8_t)v, M[c], N[c], k, tail);
      if (same)
        shift = k;
    }
    if (shift < 0)
      return failure();

    // The same enumeration says whether the clamp ever fires. Two hundred and
    // fifty-six inputs and a per-channel affine map is the whole domain, so
    // "never" here is a proof and not an argument about ranges -- and the
    // clamp is half of what this loop costs: without Zbb `maxsi` and `minsi`
    // are five instructions ([[gemmlir-two-clamps-are-a-range-check]] turns the
    // pair into one unsigned compare, but a pair is what it needs).
    const int32_t low = tail.relu ? std::max(tail.lo, 0) : tail.lo;
    bool needLow = false, needHigh = false;
    for (int64_t c = 0; c < channels; c++)
      for (int v = -128; v <= 127; v++) {
        int64_t raw =
            ((int64_t)M[c] * (int64_t)(int8_t)v + (int64_t)N[c]) >> shift;
        if (raw < low)
          needLow = true;
        if (raw > tail.hi)
          needHigh = true;
      }

    // Rewrite.
    Location loc = bn.getLoc();
    auto i32 = rewriter.getI32Type();
    auto coefTy = MemRefType::get({channels}, i32);
    auto makeGlobal = [&](ArrayRef<int32_t> data) -> Value {
      std::string name = ("__gemmlir_fixed_" + llvm::Twine((*counter)++)).str();
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<memref::GlobalOp>(
            loc, rewriter.getStringAttr(name), rewriter.getStringAttr("private"),
            TypeAttr::get(coefTy),
            DenseElementsAttr::get(RankedTensorType::get({channels}, i32), data),
            /*constant=*/rewriter.getUnitAttr(), IntegerAttr());
      }
      return rewriter.create<memref::GetGlobalOp>(loc, coefTy, name);
    };
    Value mBuf = makeGlobal(M), nBuf = makeGlobal(N);

    SmallVector<AffineMap> maps = bn.getIndexingMapsArray();
    SmallVector<AffineMap> newMaps{maps[0], maps[bArg.getArgNumber()],
                                   maps[aArg.getArgNumber()], maps.back()};
    auto fixed = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{}, ValueRange{i8src, nBuf, mBuf}, bn.getOutputs(),
        newMaps, bn.getIteratorTypesArray(),
        [&](OpBuilder &b, Location nested, ValueRange args) {
          Value q = b.create<arith::ExtSIOp>(nested, i32, args[0]);
          Value p = b.create<arith::MulIOp>(nested, q, args[2]);
          Value t = b.create<arith::AddIOp>(nested, p, args[1]);
          Value k = b.create<arith::ConstantOp>(
              nested, b.getIntegerAttr(i32, shift));
          Value v = b.create<arith::ShRSIOp>(nested, t, k);
          if (needLow)
            v = b.create<arith::MaxSIOp>(
                nested, v,
                b.create<arith::ConstantOp>(nested,
                                            b.getIntegerAttr(i32, low)));
          if (needHigh)
            v = b.create<arith::MinSIOp>(
                nested, v,
                b.create<arith::ConstantOp>(nested,
                                            b.getIntegerAttr(i32, tail.hi)));
          b.create<linalg::YieldOp>(
              nested, ValueRange{b.create<arith::TruncIOp>(
                          nested, b.getI8Type(), v)});
        });
    (void)fixed;
    rewriter.eraseOp(bn);
    return success();
  }

private:
  /// `deq` writes `f32buf` as `sitofp(i8) * scale`, elementwise and in place.
  static bool dequantizeOf(linalg::GenericOp deq, Value f32buf, Value &i8src,
                           float &scale) {
    if (!deq || deq.getInputs().size() != 1 || deq.getOutputs().size() != 1)
      return false;
    for (AffineMap m : deq.getIndexingMapsArray())
      if (!m.isIdentity())
        return false;
    Block &body = deq.getRegion().front();
    auto yield = llvm::cast<linalg::YieldOp>(body.getTerminator());
    auto mul = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mul)
      return false;
    APFloat s(0.0f);
    Value other;
    if (matchPattern(mul.getRhs(), m_ConstantFloat(&s)))
      other = mul.getLhs();
    else if (matchPattern(mul.getLhs(), m_ConstantFloat(&s)))
      other = mul.getRhs();
    else
      return false;
    auto conv = other.getDefiningOp<arith::SIToFPOp>();
    if (!conv || conv.getIn() != body.getArgument(0))
      return false;
    if (!llvm::cast<MemRefType>(deq.getInputs()[0].getType())
             .getElementType()
             .isInteger(8))
      return false;
    i8src = deq.getInputs()[0];
    scale = s.convertToFloat();
    return true;
  }

  unsigned *counter;
};

/// Once every reader of a dequantized buffer has moved to the byte it came
/// from, the loop that wrote it has nobody left.
class DropUnreadElementwise : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic->getNumResults() != 0 || generic.getOutputs().size() != 1)
      return failure();
    Value buffer = generic.getOutputs()[0];
    auto alloc = buffer.getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();
    SmallVector<Operation *> deallocs;
    for (Operation *user : buffer.getUsers()) {
      if (user == generic)
        continue;
      if (llvm::isa<memref::DeallocOp>(user)) {
        deallocs.push_back(user);
        continue;
      }
      return failure();
    }
    rewriter.eraseOp(generic);
    for (Operation *d : deallocs)
      rewriter.eraseOp(d);
    rewriter.eraseOp(alloc);
    return success();
  }
};

class BatchNormInFixedPoint_Pass
    : public impl::BatchNormInFixedPointBase<BatchNormInFixedPoint_Pass> {
public:
  using impl::BatchNormInFixedPointBase<
      BatchNormInFixedPoint_Pass>::BatchNormInFixedPointBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, arith::ArithDialect,
                    math::MathDialect>();
  }

  void runOnOperation() final {
    unsigned counter = 0;
    RewritePatternSet patterns(&getContext());
    patterns.add<BatchNormInFixedPoint>(&getContext(), &counter);
    patterns.add<DropUnreadElementwise>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
