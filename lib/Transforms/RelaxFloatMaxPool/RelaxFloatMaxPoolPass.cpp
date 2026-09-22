//===- RelaxFloatMaxPoolPass.cpp ---------------------------------*- C++ -*-===//
//
// A float max-pool whose answer becomes an integer does not need IEEE maximum.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_RELAXFLOATMAXPOOL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// Every path from `v` reaches an `arith.fptosi`/`fptoui` through operations
/// that **propagate a NaN**.
///
/// The list is not the usual "arithmetic on the way to an integer". It is
/// exactly the operations for which `f(NaN) = NaN`, because that is what makes
/// the rewrite a refinement: the two maxima differ only by producing a NaN
/// where the other produces a number, and a NaN is only unobservable if it
/// survives to the conversion, where it is poison.
///
/// So `arith.cmpf` is **not** here even though its result is an integer: it
/// answers `false` for a NaN and something else for a number, and that answer
/// is defined, not poison. Nor are `arith.maxnumf`/`minnumf`, which are the
/// relu `--select-to-minmax` leaves behind: `maxnumf(NaN, 0.0)` is `0.0`, so a
/// relu **erases** the NaN and the difference becomes an ordinary wrong number.
///
/// `arith.select` is here: its condition is an `i1` that cannot be derived from
/// this value without an `arith.cmpf`, which is refused, so either the value is
/// selected -- and the NaN goes on -- or the result does not depend on it.
static bool reachesOnlyIntegerConversion(Value v, unsigned depth = 0) {
  if (depth > 16)
    return false;
  for (Operation *user : v.getUsers()) {
    if (llvm::isa<arith::FPToSIOp, arith::FPToUIOp>(user))
      continue;
    if (!llvm::isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                   arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
                   arith::SelectOp, math::RoundEvenOp, math::FmaOp>(user))
      return false;
    if (user->getNumResults() != 1 ||
        !reachesOnlyIntegerConversion(user->getResult(0), depth + 1))
      return false;
  }
  return true;
}

/// `linalg.pooling_nhwc_max` on floats is an **IEEE maximum**: it propagates a
/// NaN. LLVM expands that into a pair of `feq.s`/`bnez`/`fmv.s` around the
/// comparison -- six instructions where `fmax.s` is one, on every step of every
/// window. In GoogLeNet, whose two transition pools are the last that still run
/// in f32, the NaN test alone is **3.2% of the model**.
///
/// `arith.maxnumf` hands back the operand that is not a NaN, and is one
/// `fmax.s`. The two differ **only** when an operand is a NaN, and where the
/// pool's answer reaches nothing but a conversion to an integer that difference
/// is unobservable -- `arith.fptosi` of a NaN is poison already. It is the same
/// licence `--select-to-minmax`, `--combine-channel-affine` and
/// `--saturate-constant-casts` turn on, asked here of the **buffer** the pool
/// writes rather than of a value, because the pool's answer reaches its readers
/// through memory.
///
/// This is not a trade against the dependency chain: it takes five instructions
/// out and puts none back ([[gemmlir-the-chain-not-the-count]]).
///
/// | googlenet | ms |
/// |---|---|
/// | `linalg.pooling_nhwc_max` on f32 | 453.48 |
/// | the same pool as `arith.maxnumf` | **409.31** |
///
/// Alternated three times each in one board session, and byte for byte against
/// both the runtime's CPU reference and the previous build over forty runs.
/// Every `feq.s` in the model -- 104 of them -- is gone, and the object is 373
/// instructions shorter. It is the only model in the set it reaches: everywhere
/// else the max-pools are already i8, which `--pack-int8-max-pool` packs eight
/// to a register.
///
/// **What is left of those two pools.** Their input is an Inception join --
/// four branches dequantized at four different scales into the channel slices
/// of one f32 buffer -- which is why they are in f32 at all. Give the four
/// convolutions one shared output scale and the join stays i8, the pool becomes
/// an i8 pool, and four dequantize loops, a pad copy and a requantize all go
/// with it. See [[gemmlir-branches-must-agree-on-the-scale]].
class RelaxFloatMaxPool : public OpRewritePattern<linalg::PoolingNhwcMaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNhwcMaxOp pool,
                                PatternRewriter &rewriter) const final {
    if (pool.getInputs().size() != 2 || pool.getOutputs().size() != 1 ||
        pool->getNumResults() != 0)
      return failure();
    Value src = pool.getInputs()[0], window = pool.getInputs()[1];
    Value out = pool.getOutputs()[0];
    auto srcTy = llvm::dyn_cast<MemRefType>(src.getType());
    auto outTy = llvm::dyn_cast<MemRefType>(out.getType());
    auto winTy = llvm::dyn_cast<MemRefType>(window.getType());
    if (!srcTy || !outTy || !winTy || srcTy.getRank() != 4 ||
        outTy.getRank() != 4 || winTy.getRank() != 2)
      return failure();
    if (!llvm::isa<FloatType>(outTy.getElementType()) ||
        outTy.getElementType() != srcTy.getElementType())
      return failure();

    // Everything that reads the buffer has to turn it into an integer. A fill
    // writes it, a deallocation ends it, and this pool is the writer; anything
    // else that reads it has to be a region whose matching argument reaches
    // nothing but a conversion.
    for (Operation *user : out.getUsers()) {
      if (user == pool.getOperation() ||
          llvm::isa<memref::DeallocOp>(user) ||
          llvm::isa<linalg::FillOp>(user))
        continue;
      auto generic = llvm::dyn_cast<linalg::GenericOp>(user);
      if (!generic || generic.getRegion().getBlocks().size() != 1)
        return failure();
      bool ok = true;
      for (OpOperand *in : generic.getDpsInputOperands()) {
        if (in->get() != out)
          continue;
        BlockArgument arg =
            generic.getRegion().front().getArgument(in->getOperandNumber());
        if (!reachesOnlyIntegerConversion(arg))
          ok = false;
      }
      // Reading it as an output would mean it is written again, which the
      // check above says nothing about.
      if (llvm::is_contained(generic.getOutputs(), out))
        ok = false;
      if (!ok)
        return failure();
    }

    auto pair = [](DenseIntElementsAttr a, unsigned i) -> int64_t {
      return (*(a.value_begin<APInt>() + i)).getSExtValue();
    };
    auto strides = pool.getStrides(), dilations = pool.getDilations();
    if (!strides || !dilations || strides.getNumElements() != 2 ||
        dilations.getNumElements() != 2)
      return failure();
    int64_t sh = pair(strides, 0), sw = pair(strides, 1);
    int64_t dh = pair(dilations, 0), dw = pair(dilations, 1);

    MLIRContext *ctx = rewriter.getContext();
    AffineExpr n, oh, ow, c, kh, kw;
    bindDims(ctx, n, oh, ow, c, kh, kw);
    SmallVector<AffineMap> maps{
        AffineMap::get(6, 0, {n, oh * sh + kh * dh, ow * sw + kw * dw, c}, ctx),
        AffineMap::get(6, 0, {kh, kw}, ctx),
        AffineMap::get(6, 0, {n, oh, ow, c}, ctx)};
    SmallVector<utils::IteratorType> iters(4, utils::IteratorType::parallel);
    iters.push_back(utils::IteratorType::reduction);
    iters.push_back(utils::IteratorType::reduction);

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        pool, TypeRange{}, ValueRange{src, window}, ValueRange{out}, maps, iters,
        [](OpBuilder &b, Location loc, ValueRange args) {
          // args = (element, window, accumulator); the window is shape only.
          Value m = b.create<arith::MaxNumFOp>(loc, args[2], args[0]);
          b.create<linalg::YieldOp>(loc, m);
        });
    return success();
  }
};

class RelaxFloatMaxPoolPass_
    : public impl::RelaxFloatMaxPoolBase<RelaxFloatMaxPoolPass_> {
public:
  using impl::RelaxFloatMaxPoolBase<
      RelaxFloatMaxPoolPass_>::RelaxFloatMaxPoolBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, arith::ArithDialect,
                    math::MathDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RelaxFloatMaxPool>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
