//===- TableForInt8ElementwisePass.cpp ---------------------------*- C++ -*-===//
//
// An elementwise function of one i8 has 256 answers. Look them up.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

#include "Gemmlir/GemmlirPasses.h"

#include <cmath>

namespace mlir::gemmlir {

#define GEN_PASS_DEF_TABLEFORINT8ELEMENTWISE
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// One value in the body, evaluated. Floats are kept at the width the operation
/// uses -- `expf` and not `exp` -- because the runtime calls the float one.
struct Scalar {
  bool isFloat = true;
  float f = 0.0f;
  int64_t i = 0;
};

/// Evaluate `v` for one input. Returns nothing for anything not on the list:
/// the table has to be what the loop would have computed, so a guess is worse
/// than declining.
static std::optional<Scalar> evaluate(Value v, Value arg, Scalar in,
                                      DenseMap<Value, Scalar> &memo,
                                      unsigned depth = 0) {
  if (v == arg)
    return in;
  if (depth > 64)
    return std::nullopt;
  auto found = memo.find(v);
  if (found != memo.end())
    return found->second;

  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;

  // A value from outside the region is a constant or it is not usable.
  llvm::APFloat cf(0.0f);
  llvm::APInt ci;
  if (matchPattern(v, m_ConstantFloat(&cf))) {
    // Only an f32 one. Everything below evaluates in `float`, so a constant in
    // a wider type would be a different computation from the loop's -- and
    // `convertToFloat` asserts on it outright, which is how DenseNet's batch
    // norm crashed the pass. Its epsilon reaches the body as an f64 constant
    // with a `truncf`, and that shape is handled just below instead.
    if (!llvm::APFloat::getSizeInBits(cf.getSemantics()) ||
        &cf.getSemantics() != &llvm::APFloat::IEEEsingle())
      return std::nullopt;
    Scalar s{true, cf.convertToFloat(), 0};
    memo[v] = s;
    return s;
  }
  if (matchPattern(v, m_ConstantInt(&ci))) {
    Scalar s{false, 0.0f, ci.getSExtValue()};
    memo[v] = s;
    return s;
  }

  auto operand = [&](unsigned k) {
    return evaluate(def->getOperand(k), arg, in, memo, depth + 1);
  };
  auto give = [&](Scalar s) -> std::optional<Scalar> {
    memo[v] = s;
    return s;
  };

  // `truncf` of a wide constant. `--fold-batch-norm` leaves the epsilon as an
  // f64 `arith.constant` outside the region and a `truncf` to f32 inside it, and
  // rounding the f64 value once to f32 is exactly what that pair computes.
  // Read from the constant rather than through `operand(0)`, which declines a
  // non-f32 constant on purpose.
  if (auto trunc = llvm::dyn_cast<arith::TruncFOp>(def)) {
    llvm::APFloat wide(0.0);
    if (!getElementTypeOrSelf(trunc.getType()).isF32() ||
        !matchPattern(trunc.getIn(), m_ConstantFloat(&wide)))
      return std::nullopt;
    bool lost = false;
    wide.convert(llvm::APFloat::IEEEsingle(),
                 llvm::APFloat::rmNearestTiesToEven, &lost);
    return give(Scalar{true, wide.convertToFloat(), 0});
  }

  if (def->getNumOperands() == 1) {
    std::optional<Scalar> a = operand(0);
    if (!a)
      return std::nullopt;
    if (llvm::isa<arith::SIToFPOp>(def))
      return give({true, static_cast<float>(a->i), 0});
    if (llvm::isa<arith::FPToSIOp>(def)) {
      // What `fcvt.w.s` does: saturate rather than wrap, which is also what
      // every quantization tail here relies on.
      float x = a->f;
      int64_t r;
      if (std::isnan(x))
        r = 0;
      else if (x >= 2147483647.0f)
        r = 2147483647;
      else if (x <= -2147483648.0f)
        r = -2147483648LL;
      else
        r = static_cast<int64_t>(x);
      return give({false, 0.0f, r});
    }
    if (llvm::isa<arith::ExtSIOp>(def))
      return give({false, 0.0f, a->i});
    if (auto trunc = llvm::dyn_cast<arith::TruncIOp>(def)) {
      unsigned w = trunc.getType().getIntOrFloatBitWidth();
      return give({false, 0.0f,
                   llvm::APInt(64, a->i).trunc(w).sext(64).getSExtValue()});
    }
    if (llvm::isa<arith::NegFOp>(def))
      return give({true, -a->f, 0});
    if (llvm::isa<math::ExpOp>(def))
      return give({true, expf(a->f), 0});
    if (llvm::isa<math::LogOp>(def))
      return give({true, logf(a->f), 0});
    if (llvm::isa<math::SqrtOp>(def))
      return give({true, sqrtf(a->f), 0});
    if (llvm::isa<math::TanhOp>(def))
      return give({true, tanhf(a->f), 0});
    if (llvm::isa<math::ErfOp>(def))
      return give({true, erff(a->f), 0});
    if (llvm::isa<math::AbsFOp>(def))
      return give({true, fabsf(a->f), 0});
    if (llvm::isa<math::RoundEvenOp>(def))
      return give({true, nearbyintf(a->f), 0});
    if (llvm::isa<math::FloorOp>(def))
      return give({true, floorf(a->f), 0});
    if (llvm::isa<math::CeilOp>(def))
      return give({true, ceilf(a->f), 0});
    return std::nullopt;
  }

  if (def->getNumOperands() == 2) {
    std::optional<Scalar> a = operand(0), b = operand(1);
    if (!a || !b)
      return std::nullopt;
    if (llvm::isa<arith::AddFOp>(def))
      return give({true, a->f + b->f, 0});
    if (llvm::isa<arith::SubFOp>(def))
      return give({true, a->f - b->f, 0});
    if (llvm::isa<arith::MulFOp>(def))
      return give({true, a->f * b->f, 0});
    if (llvm::isa<arith::DivFOp>(def))
      return give({true, a->f / b->f, 0});
    if (llvm::isa<arith::MaximumFOp, arith::MaxNumFOp>(def))
      return give({true, a->f > b->f ? a->f : b->f, 0});
    if (llvm::isa<arith::MinimumFOp, arith::MinNumFOp>(def))
      return give({true, a->f < b->f ? a->f : b->f, 0});
    if (llvm::isa<arith::AddIOp>(def))
      return give({false, 0.0f, a->i + b->i});
    if (llvm::isa<arith::SubIOp>(def))
      return give({false, 0.0f, a->i - b->i});
    if (llvm::isa<arith::MulIOp>(def))
      return give({false, 0.0f, a->i * b->i});
    if (llvm::isa<arith::MaxSIOp>(def))
      return give({false, 0.0f, std::max(a->i, b->i)});
    if (llvm::isa<arith::MinSIOp>(def))
      return give({false, 0.0f, std::min(a->i, b->i)});
    if (auto cmp = llvm::dyn_cast<arith::CmpFOp>(def)) {
      bool r;
      switch (cmp.getPredicate()) {
      case arith::CmpFPredicate::OGT:
      case arith::CmpFPredicate::UGT: r = a->f > b->f; break;
      case arith::CmpFPredicate::OGE:
      case arith::CmpFPredicate::UGE: r = a->f >= b->f; break;
      case arith::CmpFPredicate::OLT:
      case arith::CmpFPredicate::ULT: r = a->f < b->f; break;
      case arith::CmpFPredicate::OLE:
      case arith::CmpFPredicate::ULE: r = a->f <= b->f; break;
      case arith::CmpFPredicate::OEQ:
      case arith::CmpFPredicate::UEQ: r = a->f == b->f; break;
      case arith::CmpFPredicate::ONE:
      case arith::CmpFPredicate::UNE: r = a->f != b->f; break;
      default: return std::nullopt;
      }
      return give({false, 0.0f, r ? 1 : 0});
    }
    return std::nullopt;
  }

  if (def->getNumOperands() == 3 && llvm::isa<arith::SelectOp>(def)) {
    std::optional<Scalar> c = operand(0), a = operand(1), b = operand(2);
    if (!c || !a || !b)
      return std::nullopt;
    return give(c->i ? *a : *b);
  }
  return std::nullopt;
}

/// An elementwise operation whose only varying input is an **i8** has 256
/// answers, and they can all be worked out at compile time.
///
/// This is what a requantization in front of an activation buys beyond the
/// offload it was put there for. EfficientNet's SiLU -- `x * sigmoid(x)`, seven
/// operations including a `math.exp` at about 65 cycles -- reads the i8 that
/// `--quantize-unfoldable-tails` leaves behind, so the whole chain is a lookup
/// in a 256-entry table that fits in a cache line four times over.
///
/// Exact by construction: the table is what the loop computes, entry by entry,
/// with `expf` and not `exp` because that is what the loop would have called.
/// Both runtimes read the same table, so they still agree byte for byte.
/// An elementwise chain below an **i8** has 256 answers, and they can all be
/// worked out at compile time.
///
/// This is what a requantization in front of an activation buys beyond the
/// offload it was put there for. EfficientNet's SiLU -- `x * sigmoid(x)`, seven
/// operations including a `math.exp` at about 65 cycles -- sits directly under
/// the i8 `--quantize-unfoldable-tails` produces, so the whole chain is a lookup
/// in a 256-entry table that fits in four cache lines.
///
/// The anchor is a value *inside* the body, not the operation's input: fusion
/// has already put the quantization, the dequantization and the activation in
/// one region, so the byte never reaches memory. Everything below it is
/// replaced; everything above it stays where it is.
///
/// Exact by construction: the table is what the loop computes, entry by entry,
/// with `expf` and not `exp` because that is what the loop would have called.
/// Both runtimes read the same table, so they still agree byte for byte.
class TableOfOneByte : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.hasPureBufferSemantics() || generic.getOutputs().size() != 1)
      return failure();
    if (generic.getRegion().getBlocks().size() != 1)
      return failure();
    Block &body = generic.getRegion().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    for (Operation &op : body.without_terminator()) {
      // A table this pass has already put in is a load from a constant global,
      // which is pure in every way that matters here -- and refusing it is what
      // limited a region to **one** table. An LSTM's gate region needs three.
      if (llvm::isa<memref::GetGlobalOp>(&op))
        continue;
      if (auto load = llvm::dyn_cast<memref::LoadOp>(&op)) {
        if (load.getMemRef().getDefiningOp<memref::GetGlobalOp>())
          continue;
        return failure();
      }
      if (!isPure(&op) || llvm::isa<linalg::IndexOp>(&op))
        return failure();
    }

    // Every i8 the answer could depend on only through constants. The
    // operation's own inputs count: an activation that reads an i8 straight out
    // of memory -- ConvNeXt's GELU, sitting under a `matmul_i8_scale` -- has no
    // i8 *operation* in its body at all, and looking only at those found
    // nothing. All of them, not just the first: an LSTM's gates arrive as one
    // region reading **three** i8 slices of the same buffer.
    SmallVector<Value> candidates;
    for (unsigned i = 0, e = generic.getInputs().size(); i < e; i++)
      if (body.getArgument(i).getType().isInteger(8))
        candidates.push_back(body.getArgument(i));
    for (Operation &op : body.without_terminator())
      if (op.getNumResults() == 1 && op.getResult(0).getType().isInteger(8))
        candidates.push_back(op.getResult(0));

    // How many operations a value is worth, and whether any of them is the kind
    // that makes a table pay for itself.
    auto chainSize = [&](Value v, Value anchor, bool &transcendental) {
      SmallVector<Value> work{v};
      llvm::DenseSet<Operation *> seen;
      unsigned n = 0;
      while (!work.empty()) {
        Value w = work.pop_back_val();
        if (w == anchor)
          continue;
        Operation *def = w.getDefiningOp();
        if (!def || def->getBlock() != &body)
          continue;
        if (!seen.insert(def).second)
          continue;
        n++;
        if (llvm::isa<math::ExpOp, math::LogOp, math::SqrtOp, math::TanhOp,
                      math::ErfOp, math::PowFOp>(def))
          transcendental = true;
        for (Value o : def->getOperands())
          work.push_back(o);
      }
      return n;
    };

    // The **largest** value in the body that is a function of one byte, which
    // need not be the one the region yields. An LSTM's gate region computes
    // `sigmoid(f) * c + sigmoid(i) * tanh(g)`: the yielded value depends on
    // three bytes at once, but each of the three activations inside it depends
    // on exactly one, and each of those is 256 answers.
    Value anchor, target;
    unsigned best = 0;
    for (Value candidate : candidates) {
      Operation *from = candidate.getDefiningOp();
      for (Operation *n = from ? from->getNextNode() : &body.front(); n;
           n = n->getNextNode()) {
        if (llvm::isa<linalg::YieldOp>(n) || n->getNumResults() != 1)
          continue;
        Value v = n->getResult(0);
        Type t = v.getType();
        if (!t.isF32() && !t.isInteger(8) && !t.isInteger(32))
          continue;
        // The chain a previous round replaced is still sitting in the region
        // until the next canonicalization, and it still evaluates. Tabling it
        // again would go round for as long as the driver allowed.
        if (v.use_empty())
          continue;
        DenseMap<Value, Scalar> probe;
        if (!evaluate(v, candidate, Scalar{false, 0.0f, 0}, probe))
          continue;
        // A table is a kilobyte and a load; the arithmetic it replaces has to be
        // worth more than that. Measured: tabling every three-operation chain
        // below a byte put 130 tables in EfficientNet -- 130 KB competing for a
        // small L1 -- and the model went **1115 to 1214 ms**. A transcendental
        // is what makes it pay: one `math.exp` is about 65 cycles on its own.
        bool transcendental = false;
        unsigned below = chainSize(v, candidate, transcendental);
        if (below < 3 || !transcendental || below <= best)
          continue;
        anchor = candidate;
        target = v;
        best = below;
      }
    }
    if (!anchor)
      return failure();
    Type resultTy = target.getType();

    // Every answer, worked out here.
    SmallVector<Attribute> entries;
    entries.reserve(256);
    for (int v = -128; v <= 127; v++) {
      DenseMap<Value, Scalar> memo;
      std::optional<Scalar> got =
          evaluate(target, anchor, Scalar{false, 0.0f, v}, memo);
      if (!got)
        return failure();
      if (resultTy.isF32()) {
        if (!got->isFloat || !std::isfinite(got->f))
          return failure();
        entries.push_back(rewriter.getF32FloatAttr(got->f));
      } else {
        if (got->isFloat)
          return failure();
        entries.push_back(rewriter.getIntegerAttr(resultTy, got->i));
      }
    }

    auto module = generic->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto tableTy = MemRefType::get({256}, resultTy);
    std::string name;
    {
      unsigned n = 0;
      do {
        name = ("__gemmlir_table_" + llvm::Twine(n++)).str();
      } while (module.lookupSymbol(name));
    }
    Location loc = generic.getLoc();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<memref::GlobalOp>(
          loc, StringRef(name), rewriter.getStringAttr("private"), tableTy,
          DenseElementsAttr::get(RankedTensorType::get({256}, resultTy),
                                 entries),
          /*constant=*/true, /*alignment=*/rewriter.getI64IntegerAttr(64));
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(generic);
    Value table = rewriter.create<memref::GetGlobalOp>(loc, tableTy, name);

    // Right after the value it replaces, so every use of it is below.
    rewriter.setInsertionPointAfter(target.getDefiningOp());
    Value wide = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(),
                                                 anchor);
    Value bias = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(128));
    Value slot = rewriter.create<arith::AddIOp>(loc, wide, bias);
    Value idx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                    slot);
    Value got = rewriter.create<memref::LoadOp>(loc, table, ValueRange{idx});
    rewriter.replaceAllUsesWith(target, got);
    return success();
  }
};

class TableForInt8Elementwise
    : public impl::TableForInt8ElementwiseBase<TableForInt8Elementwise> {
public:
  using impl::TableForInt8ElementwiseBase<
      TableForInt8Elementwise>::TableForInt8ElementwiseBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TableOfOneByte>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
