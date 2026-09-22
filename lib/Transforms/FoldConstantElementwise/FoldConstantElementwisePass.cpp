//===- FoldConstantElementwisePass.cpp ---------------------------*- C++ -*-===//
//
// A loop whose every input is a constant is a constant.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

#include <cmath>

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FOLDCONSTANTELEMENTWISE
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// Every value the evaluator carries is a float held as a `double`; an `f32`
/// one is rounded back through `float` after each step, so the bits are the
/// ones the board would have produced.
struct Num {
  double v;
  bool wide; // f64 rather than f32
};

static double narrow(double v, bool wide) {
  return wide ? v : (double)(float)v;
}

/// The whole evaluator. Anything not listed makes the pattern fail rather than
/// guess -- a wrong constant is silent, and every operation here has to round
/// exactly the way the target does.
///
/// `math.rsqrt` is spelled the way `--math-expand-ops=ops=rsqrt` spells it in
/// LOWER, `1.0 / sqrt(x)`, because that is the code this replaces. Both halves
/// are correctly rounded in IEEE-754, on this host and on the board, so the
/// answer is the same bit pattern.
static bool evalOp(Operation *op, SmallVectorImpl<Num> &env,
                   DenseMap<Value, unsigned> &slot);

static bool isFloat(Type t) { return t.isF32() || t.isF64(); }

/// A per-channel loop that reads nothing but constants.
///
/// `--hoist-invariant-reciprocal` and `--combine-channel-affine` both *create*
/// one: a batch norm with no contraction to fold into keeps its parameters, and
/// those two passes lift `rsqrt(var[c] + eps)` and the affine that follows it
/// out of the element loop and into a loop of their own, one step per channel.
/// The parameters are `memref.global`s, so the loop computes the same numbers
/// on every inference -- 31,616 of them in DenseNet-121, and `fsqrt.s` plus
/// `fdiv.s` were **5.3% of that model** by program-counter sampling.
///
/// The passes that create these run in MID, long after the constant folding in
/// FRONT has finished; nothing was left to notice them. Same shape as
/// [[gemmlir-calibration-cannot-see-it]].
class FoldConstantElementwise : public OpRewritePattern<linalg::GenericOp> {
public:
  FoldConstantElementwise(MLIRContext *ctx, int64_t maxElements,
                          unsigned *counter)
      : OpRewritePattern(ctx), maxElements(maxElements), counter(counter) {}

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Buffer semantics: the answer goes in an `outs` buffer, not a result.
    if (generic->getNumResults() != 0)
      return failure();
    if (generic.getInputs().empty() || generic.getOutputs().empty())
      return failure();

    for (utils::IteratorType it : generic.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();
    for (AffineMap m : generic.getIndexingMapsArray())
      if (!m.isIdentity())
        return failure();

    auto module = generic->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    // One shape for everything: identity maps and equal shapes mean element `i`
    // of every operand belongs to step `i`.
    ArrayRef<int64_t> shape;
    int64_t count = 1;
    auto sameShape = [&](Value v) {
      auto ty = llvm::dyn_cast<MemRefType>(v.getType());
      if (!ty || !ty.hasStaticShape() || !ty.getLayout().isIdentity())
        return false;
      if (shape.empty() && count == 1) {
        shape = ty.getShape();
        count = ty.getNumElements();
        return count > 0;
      }
      return ty.getShape() == shape;
    };

    SmallVector<DenseElementsAttr> inputs;
    for (Value v : generic.getInputs()) {
      if (!sameShape(v))
        return failure();
      auto get = v.getDefiningOp<memref::GetGlobalOp>();
      if (!get)
        return failure();
      auto global = module.lookupSymbol<memref::GlobalOp>(get.getNameAttr());
      if (!global || !global.getConstant())
        return failure();
      auto init =
          llvm::dyn_cast_or_null<DenseElementsAttr>(global.getInitialValueAttr());
      if (!init || init.getNumElements() != count)
        return failure();
      inputs.push_back(init);
    }
    if (count > maxElements)
      return failure();

    // The outputs must be fresh buffers this loop is the only writer of, and
    // nothing may alias them: an operation with no memory effects at all (a
    // `memref.subview`, say) hands out a second name for the same bytes, so it
    // is refused along with everything else unrecognised.
    SmallVector<memref::AllocOp> allocs;
    for (Value v : generic.getOutputs()) {
      if (!sameShape(v))
        return failure();
      auto alloc = v.getDefiningOp<memref::AllocOp>();
      if (!alloc)
        return failure();
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (user == generic || llvm::isa<memref::DeallocOp>(user))
          continue;
        auto effects = llvm::dyn_cast<MemoryEffectOpInterface>(user);
        if (!effects)
          return failure();
        SmallVector<MemoryEffects::EffectInstance> found;
        effects.getEffectsOnValue(v, found);
        if (found.empty())
          return failure();
        for (const MemoryEffects::EffectInstance &e : found)
          if (!llvm::isa<MemoryEffects::Read>(e.getEffect()))
            return failure();
      }
      // Two outputs naming the same buffer would be replaced twice.
      if (llvm::is_contained(allocs, alloc))
        return failure();
      allocs.push_back(alloc);
    }

    Block &body = generic.getRegion().front();
    unsigned numIn = generic.getInputs().size();
    // An `outs` block argument is whatever the buffer already held, and a fresh
    // `memref.alloc` holds nothing.
    for (unsigned i = numIn; i < body.getNumArguments(); i++)
      if (!body.getArgument(i).use_empty())
        return failure();
    for (unsigned i = 0; i < numIn; i++)
      if (!isFloat(body.getArgument(i).getType()))
        return failure();

    auto yield = llvm::cast<linalg::YieldOp>(body.getTerminator());
    if (yield.getNumOperands() != generic.getOutputs().size())
      return failure();

    // Evaluate once per element.
    unsigned nOut = yield.getNumOperands();
    SmallVector<SmallVector<APFloat>> out(nOut);
    DenseMap<Value, unsigned> slot;
    for (Block::BlockArgListType::iterator it = body.args_begin();
         it != body.args_end(); ++it)
      slot[*it] = slot.size();
    unsigned nextSlot = body.getNumArguments();
    for (Operation &op : body.without_terminator())
      for (Value r : op.getResults())
        slot[r] = nextSlot++;

    // The epsilon of a batch norm is an `arith.constant` *outside* the region
    // with an `arith.truncf` in, so an operand that is not a block argument and
    // not another body result is the rule rather than the exception. It has to
    // be a float constant all the same.
    SmallVector<std::tuple<unsigned, double, bool>> outer;
    for (Operation &op : body.without_terminator())
      for (Value v : op.getOperands()) {
        if (slot.count(v))
          continue;
        FloatAttr f;
        if (!isFloat(v.getType()) || !matchPattern(v, m_Constant(&f)))
          return failure();
        slot[v] = nextSlot++;
        outer.push_back({slot[v], f.getValue().convertToDouble(),
                         v.getType().isF64()});
      }

    SmallVector<Num> env(nextSlot, Num{0.0, false});
    for (const std::tuple<unsigned, double, bool> &o : outer)
      env[std::get<0>(o)] = Num{std::get<1>(o), std::get<2>(o)};
    for (int64_t i = 0; i < count; i++) {
      for (unsigned j = 0; j < numIn; j++) {
        APFloat f = inputs[j].getValues<APFloat>()[i];
        bool wide = body.getArgument(j).getType().isF64();
        env[j] = Num{f.convertToDouble(), wide};
      }
      for (Operation &op : body.without_terminator())
        if (!evalOp(&op, env, slot))
          return failure();
      for (unsigned j = 0; j < nOut; j++) {
        Value v = yield.getOperand(j);
        auto ty = llvm::dyn_cast<MemRefType>(generic.getOutputs()[j].getType());
        if (!ty || !isFloat(ty.getElementType()) ||
            ty.getElementType() != v.getType())
          return failure();
        unsigned *s = slot.find(v) == slot.end() ? nullptr : &slot[v];
        if (!s)
          return failure();
        double d = env[*s].v;
        out[j].push_back(ty.getElementType().isF64()
                             ? APFloat(d)
                             : APFloat((float)d));
      }
    }

    // Rewrite: the loop and the buffers it filled go away, one constant global
    // per output takes their place.
    rewriter.eraseOp(generic);
    for (unsigned j = 0; j < nOut; j++) {
      memref::AllocOp alloc = allocs[j];
      auto ty = llvm::cast<MemRefType>(alloc.getType());
      std::string name =
          ("__gemmlir_folded_" + llvm::Twine((*counter)++)).str();
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<memref::GlobalOp>(
            alloc.getLoc(), rewriter.getStringAttr(name),
            rewriter.getStringAttr("private"), TypeAttr::get(ty),
            DenseElementsAttr::get(
                RankedTensorType::get(ty.getShape(), ty.getElementType()),
                out[j]),
            /*constant=*/rewriter.getUnitAttr(), alloc.getAlignmentAttr());
      }
      SmallVector<Operation *> deallocs;
      for (Operation *user : alloc->getUsers())
        if (llvm::isa<memref::DeallocOp>(user))
          deallocs.push_back(user);
      for (Operation *d : deallocs)
        rewriter.eraseOp(d);
      rewriter.setInsertionPoint(alloc);
      Value g = rewriter.create<memref::GetGlobalOp>(alloc.getLoc(), ty, name);
      rewriter.replaceOp(alloc, g);
    }
    return success();
  }

private:
  int64_t maxElements;
  unsigned *counter;
};

static bool evalOp(Operation *op, SmallVectorImpl<Num> &env,
                   DenseMap<Value, unsigned> &slot) {
  auto get = [&](Value v) -> Num * {
    auto it = slot.find(v);
    return it == slot.end() ? nullptr : &env[it->second];
  };
  auto put = [&](Value v, double d) {
    Num *n = get(v);
    n->wide = v.getType().isF64();
    n->v = narrow(d, n->wide);
  };
  if (!llvm::all_of(op->getResults(), [&](Value v) {
        return isFloat(v.getType()) && get(v);
      }))
    return false;
  for (Value v : op->getOperands())
    if (!get(v))
      return false;

  if (auto c = llvm::dyn_cast<arith::ConstantOp>(op)) {
    auto f = llvm::dyn_cast<FloatAttr>(c.getValue());
    if (!f)
      return false;
    put(c.getResult(), f.getValue().convertToDouble());
    return true;
  }
  auto bin = [&](Value r, Value a, Value b, char what) {
    double x = get(a)->v, y = get(b)->v;
    bool wide = r.getType().isF64();
    double z;
    if (wide) {
      switch (what) {
      case '+': z = x + y; break;
      case '-': z = x - y; break;
      case '*': z = x * y; break;
      default:  z = x / y; break;
      }
    } else {
      float fx = (float)x, fy = (float)y, fz;
      switch (what) {
      case '+': fz = fx + fy; break;
      case '-': fz = fx - fy; break;
      case '*': fz = fx * fy; break;
      default:  fz = fx / fy; break;
      }
      z = (double)fz;
    }
    put(r, z);
  };
  if (auto o = llvm::dyn_cast<arith::AddFOp>(op)) {
    bin(o.getResult(), o.getLhs(), o.getRhs(), '+');
    return true;
  }
  if (auto o = llvm::dyn_cast<arith::SubFOp>(op)) {
    bin(o.getResult(), o.getLhs(), o.getRhs(), '-');
    return true;
  }
  if (auto o = llvm::dyn_cast<arith::MulFOp>(op)) {
    bin(o.getResult(), o.getLhs(), o.getRhs(), '*');
    return true;
  }
  if (auto o = llvm::dyn_cast<arith::DivFOp>(op)) {
    bin(o.getResult(), o.getLhs(), o.getRhs(), '/');
    return true;
  }
  if (auto o = llvm::dyn_cast<arith::NegFOp>(op)) {
    put(o.getResult(), -get(o.getOperand())->v);
    return true;
  }
  if (auto o = llvm::dyn_cast<arith::TruncFOp>(op)) {
    if (!o.getIn().getType().isF64() || !o.getType().isF32())
      return false;
    put(o.getResult(), get(o.getIn())->v);
    return true;
  }
  if (auto o = llvm::dyn_cast<arith::ExtFOp>(op)) {
    if (!o.getIn().getType().isF32() || !o.getType().isF64())
      return false;
    put(o.getResult(), get(o.getIn())->v);
    return true;
  }
  if (auto o = llvm::dyn_cast<math::SqrtOp>(op)) {
    double x = get(o.getOperand())->v;
    put(o.getResult(),
        o.getType().isF64() ? std::sqrt(x) : (double)std::sqrt((float)x));
    return true;
  }
  if (auto o = llvm::dyn_cast<math::RsqrtOp>(op)) {
    double x = get(o.getOperand())->v;
    if (o.getType().isF64())
      put(o.getResult(), 1.0 / std::sqrt(x));
    else
      put(o.getResult(), (double)(1.0f / std::sqrt((float)x)));
    return true;
  }
  return false;
}

class FoldConstantElementwise_Pass
    : public impl::FoldConstantElementwiseBase<FoldConstantElementwise_Pass> {
public:
  using impl::FoldConstantElementwiseBase<
      FoldConstantElementwise_Pass>::FoldConstantElementwiseBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, arith::ArithDialect,
                    math::MathDialect>();
  }

  void runOnOperation() final {
    unsigned counter = 0;
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldConstantElementwise>(&getContext(), maxElements, &counter);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
