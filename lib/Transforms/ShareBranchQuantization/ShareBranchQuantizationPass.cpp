//===- ShareBranchQuantizationPass.cpp ----------------------*- C++ -*-===//
//
// Quantizes a branching activation once instead of once per consumer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_SHAREBRANCHQUANTIZATION
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// What a frontend puts between an activation and the convolution that reads
/// it: the layout the convolution wants, the border it reads outside the image,
/// and -- where the channels are split, as ShuffleNet's unit splits them -- the
/// part of them this branch is for. None of them looks at a value, so all of
/// them can run on the quantized one.
/// The permutation a relayout applies, whichever way it is written.
///
/// `linalg.transpose` is the named form. torch-mlir writes the *same thing* as
/// a `linalg.generic` whose body yields its input unchanged and whose maps put
/// the permutation on the **write**: `out[d0, d2, d1, d3] = in[d0, d1, d2, d3]`.
/// A transformer's attention reaches the QKV split through three of those, and
/// not recognising them is what stopped the branch walk one step below the
/// split -- so the three projections each quantized their own slice and the
/// contraction above them never folded. On `vit_tiny` that is **13.4%** of
/// everything the accelerator moves, the largest unfolded tail in the set.
///
/// Returned in `linalg.transpose`'s convention: result dimension `i` reads the
/// input's `perm[i]`.
static std::optional<SmallVector<int64_t>> relayoutPermutation(Operation *op) {
  if (auto transpose = dyn_cast<linalg::TransposeOp>(op))
    return SmallVector<int64_t>(transpose.getPermutation());
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1 || generic->getNumResults() != 1)
    return std::nullopt;
  if (!llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType it) {
                      return it == utils::IteratorType::parallel;
                    }))
    return std::nullopt;
  Block &body = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != body.getArgument(0))
    return std::nullopt;
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isPermutation())
    return std::nullopt;
  // `out[maps[1](d)] = in[d]`, so out index j reads in[inverse(maps[1])(j)].
  AffineMap inverse = inversePermutation(maps[1]);
  if (!inverse)
    return std::nullopt;
  SmallVector<int64_t> perm;
  for (AffineExpr e : inverse.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(e);
    if (!dim)
      return std::nullopt;
    perm.push_back(dim.getPosition());
  }
  return perm;
}

bool isLayoutOnly(Operation *op) {
  // A reshape is on the list because a grouped convolution arrives as one:
  // `--split-grouped-conv` collapses the 5-D (N, G, C/G, H, W) form back to
  // NCHW before it relayouts, and a symmetric quantization commutes with that
  // as it does with any of these.
  if (isa<tensor::PadOp, tensor::ExtractSliceOp, tensor::CollapseShapeOp,
          tensor::ExpandShapeOp>(op))
    return true;
  return relayoutPermutation(op).has_value();
}

/// Zero is the only padding value a symmetric quantization maps to itself, and
/// the only border `tiled_conv_auto` can produce.
bool padsWithZero(tensor::PadOp pad) {
  auto yield = dyn_cast<tensor::YieldOp>(pad.getRegion().front().getTerminator());
  if (!yield)
    return false;
  APFloat value(0.0);
  return matchPattern(yield.getValue(), m_ConstantFloat(&value)) && value.isZero();
}

/// Walks from a `quant.qcast` up through layout operations to the value the
/// activation really is. Every step has to be the only user of what it reads:
/// moving the chain into i8 while an f32 copy of it stayed behind would cost
/// more than it saves.
Value branchRootOf(quant::QuantizeCastOp qcast,
                   SmallVectorImpl<Operation *> &chain) {
  Value v = qcast.getInput();
  while (Operation *def = v.getDefiningOp()) {
    if (!isLayoutOnly(def) || !def->hasOneUse())
      break;
    if (auto pad = dyn_cast<tensor::PadOp>(def))
      if (!padsWithZero(pad))
        break;
    chain.push_back(def);
    v = def->getOperand(0);
  }
  return v;
}

/// The dequantization this pass writes at a branch, possibly relaid out: a
/// scaled integer. Matching one as a branch root would re-share what is already
/// shared, and this is what terminates the rewrite.
bool isSharedDequantization(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (!relayoutPermutation(def))
      break;
    v = def->getOperand(0);
  }
  auto mul = v.getDefiningOp<arith::MulFOp>();
  return mul && mul.getLhs().getDefiningOp<arith::SIToFPOp>();
}

/// `after` applied to the result of `first`: result dimension i is `first`'s
/// dimension after[i], which is the input's dimension first[after[i]].
SmallVector<int64_t> compose(ArrayRef<int64_t> first, ArrayRef<int64_t> after) {
  SmallVector<int64_t> out(after.size());
  for (size_t i = 0; i < after.size(); i++)
    out[i] = first[after[i]];
  return out;
}

bool isIdentity(ArrayRef<int64_t> perm) {
  for (size_t i = 0; i < perm.size(); i++)
    if (perm[i] != static_cast<int64_t>(i))
      return false;
  return true;
}

/// A per-dimension quantity carried through a transpose: the result's dimension
/// i is the input's dimension perm[i], so it takes that dimension's value.
SmallVector<OpFoldResult> permute(ArrayRef<OpFoldResult> values,
                                  ArrayRef<int64_t> perm) {
  SmallVector<OpFoldResult> out;
  for (int64_t p : perm)
    out.push_back(values[p]);
  return out;
}

/// `--force-quantized-matmul` quantizes each contraction's operands where the
/// contraction reads them. At a branch that is the wrong place: the activation
/// stays f32 because the other consumer wants it that way, so the producing
/// contraction's tail is f32 too and never ends in the requantization that
/// `--convert-linalg-to-gemmlir` folds.
///
/// This quantizes the branch once and hands the other consumers the
/// dequantization of that one value, which is what a quantized residual network
/// does anyway -- and is the precondition for reaching `resadd_i8`.
///
/// The quantization is put in the *convolution's* layout, not the branch's. The
/// transposes a frontend leaves between the two are hoisted in front of it, so
/// the quantization comes to sit on a transpose of the tail and the existing
/// absorb-and-fuse machinery can pull it into the tail -- which is what makes
/// the tail end in the convolution's own layout, the last condition
/// `matchRequantize` puts on it. Left in the branch's layout it fuses just as
/// well and still does not fold.
class ShareAtBranch : public OpRewritePattern<quant::QuantizeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quant::QuantizeCastOp qcast,
                                PatternRewriter &rewriter) const final {
    SmallVector<Operation *> chain;
    Value root = branchRootOf(qcast, chain);
    if (root.hasOneUse() || isSharedDequantization(root))
      return failure();
    // Two reasons to quantize at a branch. The first is the one this was
    // written for: the value comes off a quantized contraction, whose tail then
    // ends in a requantization and folds.
    //
    // The second is that **some consumer is quantizing this value anyway** --
    // which is what being anchored on a `quant.qcast` means -- so doing it once
    // here costs that consumer nothing and turns every other consumer's read of
    // an f32 tensor into a read of an i8 one. EfficientNet's sixteen
    // squeeze-excitation blocks are that shape and the first reason misses
    // them: the depthwise convolution below them already writes i8, so the
    // branch is a *SiLU's* output rather than a contraction's. Taking them is
    // worth **550.8 -> 400.5 ms** on EfficientNet and **273.3 -> 150.7** on
    // RegNet, byte-identical to the CPU reference over forty runs each.
    //
    // The second needs a guard the first does not, and it took three wrong
    // ones to find it. What makes sharing free is not what the *other*
    // consumers do with the value -- it is that the branches **already agree**
    // on the quantization being hoisted. An LSTM's input is sixteen timesteps
    // sliced out of one tensor, each calibrated on its own, with scales from
    // 0.0128 to 0.0248; hoisting one of them over all sixteen makes the widest
    // timestep saturate at the narrowest one's range, and that is the whole of
    // 0.0066 -> 0.0284 relative L2. EfficientNet's squeeze-excitation gate has
    // one quantized branch and so has nothing to disagree with.
    //
    // The three that failed, each measured: a 4096-element size floor (reverts
    // ShuffleNet's 10 ms and costs RegNet 76 of its 128); "the other consumers
    // are quantized within a few steps" on any path; and the same on all paths
    // -- both of which left the LSTM at 0.0284, because the harm was never
    // about the other consumers at all.
    //
    // The other consumers still get a say, but a much smaller one: none of them
    // may be the *answer*. Handing a function result the dequantization of an
    // i8 where it had an f32 is a rounding nothing downstream absorbs, so every
    // path out of such a consumer has to end in a quantization of its own.
    //
    // Both of these are asked **only of the second reason**. Off a contraction
    // the prize is not a cheaper read, it is that the layer offloads at all, and
    // that is worth a scale one branch did not ask for: ShuffleNet's unit is a
    // branch whose two sides were calibrated apart, and refusing it there took
    // the probe from **2.37 to 54.43 ms**.
    auto rootTy = dyn_cast<RankedTensorType>(root.getType());
    auto qTy = dyn_cast<quant::UniformQuantizedType>(
        getElementTypeOrSelf(qcast.getType()));
    if (!rootTy || !rootTy.hasStaticShape() || !rootTy.getElementType().isF32())
      return failure();
    // The dequantization below is a plain multiply, which is only what this
    // quantization means when it is symmetric.
    if (!qTy || qTy.getZeroPoint() != 0)
      return failure();
    if (!comesOffAQuantizedContraction(root)) {
      if (!branchesAgreeOnTheQuantization(root, qTy))
        return failure();
      for (Operation *user : root.getUsers()) {
        if (llvm::is_contained(chain, user) ||
            llvm::isa<quant::QuantizeCastOp>(user))
          continue;
        if (user->getNumResults() != 1 ||
            !everyPathQuantized(user->getResult(0), 4))
          return failure();
      }
    }

    // Walk the chain the way the data does and work out, for each pad, which
    // transposes used to come after it: those move in front, so its amounts
    // move with them.
    SmallVector<Operation *> flow(chain.rbegin(), chain.rend());
    unsigned rank = rootTy.getRank();
    SmallVector<int64_t> identity(rank);
    for (unsigned i = 0; i < rank; i++)
      identity[i] = i;
    SmallVector<SmallVector<int64_t>> afterTransposes(flow.size(), identity);
    SmallVector<int64_t> suffix = identity;
    for (int j = flow.size() - 1; j >= 0; j--) {
      afterTransposes[j] = suffix;
      if (std::optional<SmallVector<int64_t>> perm =
              relayoutPermutation(flow[j])) {
        // A reshape below would have changed the rank, and this composition is
        // of permutations of one rank.
        if (perm->size() != suffix.size())
          return failure();
        suffix = compose(*perm, suffix);
      }
    }
    SmallVector<int64_t> layout = suffix; // every transpose, composed
    if (layout.size() != rank)
      return failure();

    // Only a chain the replay below can rebuild. A reshape changes the rank, so
    // the per-dimension bookkeeping above stops applying across it: it is
    // allowed only where no transpose follows it, which is how a grouped
    // convolution's collapse arrives and is the shape this was written for.
    // Anything else would be dropped silently and leave the storage cast the
    // wrong shape -- `'quant.scast' op failed to verify`, which is what a
    // squeeze-excite block's `1x64x64x32 -> 4096x32` pool produced.
    for (size_t j = 0; j < flow.size(); j++) {
      if (isa<tensor::ExtractSliceOp, tensor::PadOp>(flow[j]) ||
          relayoutPermutation(flow[j]))
        continue;
      if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(flow[j])) {
        if (!isIdentity(afterTransposes[j]))
          return failure();
        continue;
      }
      return failure();
    }

    Location loc = qcast.getLoc();
    rewriter.setInsertionPointAfterValue(root);

    // Into the convolution's layout...
    Value relaid = root;
    if (!isIdentity(layout)) {
      SmallVector<int64_t> shape;
      for (int64_t d : layout)
        shape.push_back(rootTy.getDimSize(d));
      Value init = rewriter.create<tensor::EmptyOp>(loc, shape,
                                                    rootTy.getElementType());
      relaid = rewriter.create<linalg::TransposeOp>(loc, root, init, layout)
                   ->getResult(0);
    }
    auto relaidTy = cast<RankedTensorType>(relaid.getType());

    // ...quantize once...
    Value shared = rewriter.create<quant::QuantizeCastOp>(
        loc, relaidTy.clone(qTy), relaid);
    Value storage = rewriter.create<quant::StorageCastOp>(
        loc, relaidTy.clone(qTy.getStorageType()), shared);

    // ...rebuild what is left of the chain on the storage type. Only the pads
    // are left; the transposes are the relayout above.
    Value moved = storage;
    for (size_t j = 0; j < flow.size(); j++) {
      if (auto slice = dyn_cast<tensor::ExtractSliceOp>(flow[j])) {
        auto sliceTy = cast<RankedTensorType>(slice.getResult().getType());
        if (sliceTy.getRank() != static_cast<int64_t>(afterTransposes[j].size()))
          return failure();
        SmallVector<int64_t> shape;
        for (int64_t d : afterTransposes[j])
          shape.push_back(sliceTy.getDimSize(d));
        Type elem = cast<RankedTensorType>(moved.getType()).getElementType();
        moved = rewriter.create<tensor::ExtractSliceOp>(
            loc, RankedTensorType::get(shape, elem), moved,
            permute(slice.getMixedOffsets(), afterTransposes[j]),
            permute(slice.getMixedSizes(), afterTransposes[j]),
            permute(slice.getMixedStrides(), afterTransposes[j]));
        continue;
      }
      if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(flow[j])) {
        auto ty = cast<RankedTensorType>(collapse.getResult().getType());
        Type elem = cast<RankedTensorType>(moved.getType()).getElementType();
        moved = rewriter.create<tensor::CollapseShapeOp>(
            loc, RankedTensorType::get(ty.getShape(), elem), moved,
            collapse.getReassociationIndices());
        continue;
      }
      if (auto expand = dyn_cast<tensor::ExpandShapeOp>(flow[j])) {
        auto ty = cast<RankedTensorType>(expand.getResult().getType());
        Type elem = cast<RankedTensorType>(moved.getType()).getElementType();
        moved = rewriter.create<tensor::ExpandShapeOp>(
            loc, RankedTensorType::get(ty.getShape(), elem), moved,
            expand.getReassociationIndices(), expand.getMixedOutputShape());
        continue;
      }
      auto pad = dyn_cast<tensor::PadOp>(flow[j]);
      if (!pad)
        continue;
      auto padTy = cast<RankedTensorType>(pad.getResult().getType());
      SmallVector<int64_t> shape;
      for (int64_t d : afterTransposes[j])
        shape.push_back(padTy.getDimSize(d));
      Type elem = cast<RankedTensorType>(moved.getType()).getElementType();
      Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elem));
      moved = rewriter.create<tensor::PadOp>(
          loc, RankedTensorType::get(shape, elem), moved,
          permute(pad.getMixedLowPad(), afterTransposes[j]),
          permute(pad.getMixedHighPad(), afterTransposes[j]), zero,
          /*nofold=*/false);
    }

    // ...and let the other consumers read it back. Written out as arithmetic
    // rather than as a `quant.dcast`, because the quant dialect folds
    // `dcast(qcast(x))` back to `x` -- it takes a quantization to be exact --
    // which undoes this rewrite as fast as it is made. It is what
    // `--lower-quant-ops` emits for a dcast anyway.
    Value wide = rewriter.create<arith::SIToFPOp>(loc, relaidTy, storage);
    Value scale = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(relaidTy,
                                    APFloat(static_cast<float>(qTy.getScale()))));
    Value dequantized = rewriter.create<arith::MulFOp>(loc, wide, scale);
    if (!isIdentity(layout)) {
      SmallVector<int64_t> back(rank);
      for (unsigned i = 0; i < rank; i++)
        back[layout[i]] = i;
      Value init = rewriter.create<tensor::EmptyOp>(loc, rootTy.getShape(),
                                                    rootTy.getElementType());
      dequantized =
          rewriter.create<linalg::TransposeOp>(loc, dequantized, init, back)
              ->getResult(0);
    }
    rewriter.replaceUsesWithIf(root, dequantized, [&](OpOperand &use) {
      return use.getOwner() != relaid.getDefiningOp() &&
             use.getOwner() != shared.getDefiningOp();
    });
    rewriter.replaceOpWithNewOp<quant::StorageCastOp>(qcast, qcast.getType(),
                                                      moved);
    return success();
  }

private:
  /// True if `v` is the tail of a quantized contraction: a dequantization of an
  /// i32 accumulator, reached through elementwise work. Without this the
  /// rewrite would quantize a branch no accelerator call will ever see, which
  /// is a loss of precision for nothing.
  ///
  /// The budget bounds the search, it does not express a rule -- so it has to
  /// be longer than the chains that actually occur. ResNet-18's max-pool is
  /// **seven** steps from its accumulator (residual add, relayout, tail
  /// generic, relayout, pad, pool) and at six it was refused, which left the
  /// pool's two consumers quantizing it separately: the stem convolution's tail
  /// stayed f32 and the block below ended in a requantization with three
  /// inputs, and neither convolution folded.
  /// True when every path out of this value ends in a quantization within a
  /// few steps, so nothing downstream keeps the f32 it had. A function result
  /// is the case this is for: it is not a consumer, it is the answer.
  static bool everyPathQuantized(Value v, int depth) {
    if (depth == 0 || v.use_empty())
      return false;
    for (Operation *user : v.getUsers()) {
      if (llvm::isa<quant::QuantizeCastOp>(user))
        continue;
      if ((!user->hasTrait<OpTrait::Elementwise>() && !isLayoutOnly(user)) ||
          user->getNumResults() != 1)
        return false;
      if (!everyPathQuantized(user->getResult(0), depth - 1))
        return false;
    }
    return true;
  }

  /// True when every quantization this value reaches through layout-only
  /// operations asks for `qTy`. Those are the branches the rewrite will feed
  /// from one shared quantization, so a disagreement between them is a
  /// quantization one branch did not ask for -- see the note at the guard.
  /// Consumers that are not quantizations are not part of the question: they
  /// keep reading a dequantization of the same value.
  static bool branchesAgreeOnTheQuantization(Value v,
                                             quant::UniformQuantizedType qTy,
                                             int depth = 6) {
    for (Operation *user : v.getUsers()) {
      if (auto cast = dyn_cast<quant::QuantizeCastOp>(user)) {
        if (getElementTypeOrSelf(cast.getType()) != qTy)
          return false;
        continue;
      }
      if (!isLayoutOnly(user))
        continue;
      // A chain longer than the walk above would follow anyway: refuse rather
      // than hoist over a quantization this never looked at.
      if (depth == 0 || user->getNumResults() != 1)
        return false;
      if (!branchesAgreeOnTheQuantization(user->getResult(0), qTy, depth - 1))
        return false;
    }
    return true;
  }

  static bool comesOffAQuantizedContraction(Value v, int depth = 12) {
    Operation *def = v.getDefiningOp();
    if (!def || depth == 0)
      return false;
    if (auto dcast = dyn_cast<quant::DequantizeCastOp>(def)) {
      auto qTy = dyn_cast<quant::QuantizedType>(
          getElementTypeOrSelf(dcast.getInput().getType()));
      return qTy && qTy.getStorageTypeIntegralWidth() == 32;
    }
    // Through the same layout-only operations the root walk goes through: a
    // padded convolution puts a `tensor.pad` between the contraction and the
    // branch, and it is neither a linalg operation nor elementwise, so
    // stopping here would refuse every padded convolution's branch.
    if (!isa<linalg::LinalgOp>(def) && !def->hasTrait<OpTrait::Elementwise>() &&
        !isLayoutOnly(def))
      return false;
    return llvm::any_of(def->getOperands(), [&](Value operand) {
      return comesOffAQuantizedContraction(operand, depth - 1);
    });
  }
};

class ShareBranchQuantization
    : public impl::ShareBranchQuantizationBase<ShareBranchQuantization> {
public:
  using impl::ShareBranchQuantizationBase<
      ShareBranchQuantization>::ShareBranchQuantizationBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    quant::QuantDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ShareAtBranch>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
