//===- QuantizeUnfoldableTailsPass.cpp ---------------------------*- C++ -*-===//
//
// A layer whose activation the accelerator cannot end in gets a requantization
// of its own, so the layer itself can still be offloaded.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_QUANTIZEUNFOLDABLETAILS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// A splat constant, or a per-channel broadcast of one -- the two shapes the
/// accelerator's own output pipeline can add or scale by.
static bool isPerChannelOrSplat(Value v) {
  // The layout rewrite relays a broadcast bias out like everything else, so
  // what arrives is a transpose of one. ConvNeXt's depthwise convolutions all
  // carry a bias in that shape, and reading only the transpose made every one
  // of them look like a residual add.
  for (unsigned step = 0; step < 4; step++) {
    Operation *def = v.getDefiningOp();
    if (auto transpose = llvm::dyn_cast_or_null<linalg::TransposeOp>(def))
      v = transpose.getInput();
    else if (auto collapse =
                 llvm::dyn_cast_or_null<tensor::CollapseShapeOp>(def))
      v = collapse.getSrc();
    else if (auto expand = llvm::dyn_cast_or_null<tensor::ExpandShapeOp>(def))
      v = expand.getSrc();
    else
      break;
  }
  if (matchPattern(v, m_Constant()))
    return true;
  auto generic = v.getDefiningOp<linalg::GenericOp>();
  if (!generic || generic.getInputs().size() != 1)
    return false;
  auto inTy = llvm::dyn_cast<RankedTensorType>(generic.getInputs()[0].getType());
  return inTy && inTy.getRank() == 1;
}

/// True when `user` reading `v` is something `matchRequantize` will absorb into
/// the accelerator call: a scale, a per-channel bias, a relu or a bound.
static bool absorbsIntoTheCall(Operation *user, Value v) {
  auto other = [&](unsigned i) { return user->getOperand(i) == v
                                            ? user->getOperand(1 - i)
                                            : user->getOperand(i); };
  if (user->getNumOperands() != 2 || user->getNumResults() != 1)
    return false;
  Value rest = user->getOperand(0) == v ? user->getOperand(1)
                                        : user->getOperand(0);
  (void)other;
  if (llvm::isa<arith::AddFOp>(user))
    return isPerChannelOrSplat(rest);
  if (llvm::isa<arith::MulFOp, arith::MaximumFOp, arith::MinimumFOp,
                arith::MaxNumFOp, arith::MinNumFOp>(user))
    return matchPattern(rest, m_Constant());
  if (llvm::isa<arith::DivFOp>(user))
    return user->getOperand(0) == v && matchPattern(rest, m_Constant());
  return false;
}

/// True when every operation in this body is one the accelerator's own output
/// pipeline can do. The same whitelist `--conv-to-img2col` uses, and for the
/// same reason: what stops a tail folding is a transcendental, not a relayout.
static bool bodyIsAbsorbable(Operation *op) {
  auto generic = llvm::dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return true;
  for (Operation &inner : generic.getRegion().front())
    if (!llvm::isa<arith::SIToFPOp, arith::FPToSIOp, arith::TruncIOp,
                   arith::ExtSIOp, arith::TruncFOp, arith::ExtFOp,
                   arith::MulFOp, arith::DivFOp, arith::AddFOp, arith::SubFOp,
                   arith::NegFOp, arith::AddIOp, arith::SubIOp, arith::MaxSIOp,
                   arith::MinSIOp, arith::MaximumFOp, arith::MinimumFOp,
                   arith::MaxNumFOp, arith::MinNumFOp, arith::CmpFOp,
                   arith::CmpIOp, arith::SelectOp, arith::ConstantOp,
                   math::RoundEvenOp, linalg::YieldOp>(&inner))
      return false;
  return true;
}

/// True when this map reads the whole tensor, element for element, in order.
///
/// `isIdentity()` is not the question. torch-mlir writes a batch of one as the
/// **constant 0** rather than as a loop -- `(d0, d1, d2) -> (0, d1, d2)` -- and
/// on an extent of one that is the identity. Every one of a ViT's bias adds is
/// spelled that way, so asking `isIdentity()` answered no to all of them.
static bool readsWholeTensor(AffineMap map, ShapedType type) {
  ArrayRef<int64_t> shape = type.getShape();
  if (map.getNumResults() != shape.size())
    return false;
  for (auto [r, e] : llvm::enumerate(map.getResults())) {
    if (auto dim = llvm::dyn_cast<AffineDimExpr>(e)) {
      if (dim.getPosition() != r)
        return false;
      continue;
    }
    auto cst = llvm::dyn_cast<AffineConstantExpr>(e);
    if (!cst || cst.getValue() != 0 || shape[r] != 1)
      return false;
  }
  return true;
}

/// True when this `linalg.generic` is a tail step the call absorbs.
///
/// The same question `absorbsIntoTheCall` asks, for the same tail written the
/// other way. A convolution network's bias reaches here as a tensor-level
/// `arith.addf` against a broadcast; a **transformer's** reaches here as one
/// `linalg.generic` reading the accumulator and the bias together, which has
/// three operands and no `arith` op of its own at this level, so the walk
/// stopped on it and called it the blocker. Every one of a ViT's twelve GELUs
/// was on the other side of that stop.
static bool genericAbsorbsIntoTheCall(linalg::GenericOp generic, Value v) {
  if (generic.getNumResults() != 1 || !bodyIsAbsorbable(generic))
    return false;
  if (!llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType it) {
                      return it == utils::IteratorType::parallel;
                    }))
    return false;
  unsigned loops = generic.getNumLoops();
  bool readsV = false;
  for (OpOperand *in : generic.getDpsInputOperands()) {
    AffineMap map = generic.getMatchingIndexingMap(in);
    if (in->get() == v) {
      // The walked value has to arrive whole: what the call wrote is what this
      // step reads, element for element.
      auto type = llvm::dyn_cast<ShapedType>(v.getType());
      if (!type || !readsWholeTensor(map, type))
        return false;
      readsV = true;
      continue;
    }
    // Anything else has to be what the call already carries next to its output:
    // a constant, or a value read through a map that drops a loop, which is
    // what a per-channel bias is.
    if (matchPattern(in->get(), m_Constant()))
      continue;
    if (map.getNumResults() < loops)
      continue;
    return false;
  }
  return readsV;
}

/// True when this operation ends a tail in a way the accelerator's call cannot,
/// and no other pass will rearrange into something it can.
///
/// Everything with a pass of its own is left alone -- a residual add, a pool, a
/// join, a quantization, a contraction -- because putting a requantization in
/// front of those takes work *away* from the accelerator. It cost EfficientNet
/// all nine of its `resadd_i8` the first time this fired on anything that
/// merely stopped the walk.
///
/// What is left is a `linalg.generic` the pipeline cannot do: a transcendental,
/// or a **reduction**. A layer norm's mean is a reduction, and ConvNeXt puts one
/// under every single depthwise convolution it has.
static bool stopsTheTail(Operation *op) {
  if (llvm::isa<quant::QuantizeCastOp, tensor::ConcatOp,
                tensor::InsertSliceOp, arith::AddFOp>(op))
    return false;
  auto asLinalg = llvm::dyn_cast<linalg::LinalgOp>(op);
  if (!asLinalg || linalg::isaContractionOpInterface(asLinalg))
    return false;
  if (llvm::isa<linalg::PoolingNhwcMaxOp, linalg::PoolingNchwMaxOp,
                linalg::PoolingNhwcSumOp, linalg::PoolingNchwSumOp,
                linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(op))
    return false;
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(op)) {
    bool parallel = llvm::all_of(generic.getIteratorTypesArray(),
                                 [](utils::IteratorType it) {
                                   return it == utils::IteratorType::parallel;
                                 });
    return !parallel || !bodyIsAbsorbable(generic);
  }
  // Tried and measured: treating any elementwise operation on tensors as a
  // blocker as well -- which is how ConvNeXt's GELU arrives before
  // `--convert-elementwise-to-linalg` -- cuts far more tails and is **slower**.
  // EfficientNet went 1114.08 to 1162.15 ms with the same relative L2, because
  // each cut is a requantization and most of those tails were cheap. Only a
  // region the pipeline cannot do counts.
  return false;
}

/// True when `user` joins two paths that both come from `v`.
///
/// The call writes each output element once, from one accumulator, so a tail
/// step that reads that accumulator on **two** paths at once is not something
/// its `mvout` can do -- whatever the two paths themselves contain.
///
/// A SiLU is that shape and `stopsTheTail` already catches it, but only because
/// of the `math.exp` on one of the branches. A **hard**-swish is
/// `x * clamp(x + 3, 0, 6) / 6`: every step of it is absorbable on its own, no
/// branch holds a transcendental, and it walked past every other test. Ten
/// convolutions and eight depthwise convolutions of MobileNetV3-Small kept their
/// scalar loops because of it.
static bool joinsTwoPathsFrom(Operation *user, Value v) {
  // Both spellings of the join: the `linalg.generic` torch-mlir writes for an
  // elementwise multiply of two tensors, and the tensor-level `arith` operation
  // it arrives as before `--convert-elementwise-to-linalg`.
  SmallVector<Value> inputs;
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(user))
    inputs.assign(generic.getInputs().begin(), generic.getInputs().end());
  else if (user->hasTrait<OpTrait::Elementwise>())
    inputs.assign(user->getOperands().begin(), user->getOperands().end());
  if (inputs.size() < 2)
    return false;
  // Everything `v` reaches in a few steps, not going through `user` itself.
  llvm::DenseSet<Value> from{v};
  SmallVector<Value> work{v};
  for (unsigned step = 0; step < 8 && !work.empty(); step++) {
    SmallVector<Value> next;
    for (Value w : work)
      for (Operation *u : w.getUsers())
        if (u != user && u->getNumResults() == 1 &&
            from.insert(u->getResult(0)).second)
          next.push_back(u->getResult(0));
    work = next;
  }
  unsigned reached = 0;
  for (Value in : inputs)
    if (from.contains(in))
      reached++;
  return reached >= 2;
}

/// The `gemmlir.output_scale` of the contraction this value came out of, looking
/// through the part of its tail the call itself could have done.
static std::optional<double> outputScaleBehind(Value v) {
  for (unsigned step = 0; step < 8; step++) {
    if (auto dcast = v.getDefiningOp<quant::DequantizeCastOp>()) {
      auto scast = dcast.getInput().getDefiningOp<quant::StorageCastOp>();
      if (!scast)
        return std::nullopt;
      Operation *contraction = scast.getInput().getDefiningOp();
      if (!contraction || !llvm::isa<linalg::LinalgOp>(contraction))
        return std::nullopt;
      auto attr =
          contraction->getAttrOfType<FloatAttr>("gemmlir.output_scale");
      if (!attr)
        return std::nullopt;
      double s = attr.getValueAsDouble();
      if (!(s > 0.0) || !std::isfinite(s))
        return std::nullopt;
      return s;
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      return std::nullopt;
    if (auto generic = llvm::dyn_cast<linalg::GenericOp>(def)) {
      Value up;
      for (OpOperand *in : generic.getDpsInputOperands())
        if (genericAbsorbsIntoTheCall(generic, in->get())) {
          up = in->get();
          break;
        }
      if (!up)
        return std::nullopt;
      v = up;
      continue;
    }
    if (llvm::isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

/// When `user` is the sum of `here` and **another contraction's** tail, that
/// other contraction's output scale.
///
/// Two spellings: a tensor-level `arith.addf`, and the `linalg.generic` with two
/// full-rank inputs and one `addf` in it that torch-mlir writes for the same
/// thing. An LSTM's gate is the second.
static std::optional<double> otherAccumulatorAdded(Operation *user, Value here) {
  if (llvm::isa<arith::AddFOp>(user) && user->getNumOperands() == 2) {
    Value other = user->getOperand(0) == here ? user->getOperand(1)
                                              : user->getOperand(0);
    return outputScaleBehind(other);
  }
  auto generic = llvm::dyn_cast<linalg::GenericOp>(user);
  if (!generic || generic.getNumResults() != 1 ||
      generic.getDpsInputOperands().size() != 2)
    return std::nullopt;
  if (!llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType it) {
                      return it == utils::IteratorType::parallel;
                    }))
    return std::nullopt;
  Block &body = generic.getRegion().front();
  if (body.getOperations().size() != 2)
    return std::nullopt;
  auto add = llvm::dyn_cast<arith::AddFOp>(&body.front());
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!add || !yield || yield.getOperand(0) != add.getResult())
    return std::nullopt;
  Value a = body.getArgument(0), b = body.getArgument(1);
  if (!((add.getLhs() == a && add.getRhs() == b) ||
        (add.getLhs() == b && add.getRhs() == a)))
    return std::nullopt;
  Value other;
  for (OpOperand *in : generic.getDpsInputOperands()) {
    auto type = llvm::dyn_cast<ShapedType>(in->get().getType());
    if (!type || !readsWholeTensor(generic.getMatchingIndexingMap(in), type))
      return std::nullopt;
    if (in->get() != here)
      other = in->get();
  }
  if (!other)
    return std::nullopt;
  return outputScaleBehind(other);
}

/// The operation below `v` that ends the tail, or null when nothing does.
static Operation *blockerBelow(Value v, unsigned budget) {
  for (unsigned step = 0; step < budget; step++) {
    if (!v.hasOneUse()) {
      for (Operation *u : v.getUsers())
        if (stopsTheTail(u))
          return u;
      return nullptr;
    }
    Operation *user = *v.getUsers().begin();
    if (stopsTheTail(user))
      return user;
    if (absorbsIntoTheCall(user, v)) {
      v = user->getResult(0);
      continue;
    }
    if (auto generic = llvm::dyn_cast<linalg::GenericOp>(user)) {
      if (!genericAbsorbsIntoTheCall(generic, v))
        return nullptr;
      v = generic.getResult(0);
      continue;
    }
    if (llvm::isa<linalg::TransposeOp, tensor::PadOp, tensor::ExtractSliceOp,
                  tensor::CollapseShapeOp, tensor::ExpandShapeOp>(user)) {
      v = user->getResult(0);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

/// Put a requantization where the accelerator needs one.
///
/// `conv2d_i8` and `depthwise_conv2d_i8` **write i8**: the tail has to be
/// `saturate(scale * accumulator + bias)` with at most a relu. EfficientNet's
/// activation is SiLU, `x * sigmoid(x)`, which is none of those -- so every one
/// of its sixteen depthwise convolutions stayed a scalar loop, and a depthwise
/// convolution has no matmul to fall back on the way a dense one does.
///
/// Quantizing the layer's own output first splits the tail in two: the part the
/// call can do, and the part the core does afterwards. Measured on the board at
/// 64x64, the sixteen scalar depthwise convolutions are **1749 ms of
/// EfficientNet's 2982**, and the extra round trip costs **0.00016** relative L2
/// against a model whose own quantization error is 0.0186.
///
/// The scale is the layer's *own* output range, which the calibration measures
/// separately from the next layer's input range -- there is an activation
/// between them, and it is exactly the activation that made this necessary.
/// Without `gemmlir.output_scale` there is nothing to quantize at and the tail
/// is left alone.
class QuantizeBeforeUnfoldableActivation
    : public OpRewritePattern<quant::DequantizeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quant::DequantizeCastOp dcast,
                                PatternRewriter &rewriter) const final {
    auto scast = dcast.getInput().getDefiningOp<quant::StorageCastOp>();
    if (!scast)
      return failure();
    Operation *contraction = scast.getInput().getDefiningOp();
    if (!contraction || !llvm::isa<linalg::LinalgOp>(contraction))
      return failure();
    // Tried and reverted: confining this to a **depthwise** convolution, on the
    // reasoning that a dense one can be packed as a matmul and a matmul can
    // leave its i32 accumulator for the tail to read. It is worse --
    // EfficientNet **1114 -> 1372 ms** -- because two of its dense
    // convolutions were folding into `conv2d_i8` *because* of a cut, and a
    // packed matmul is the slower home for them. A cut is not only for what has
    // nowhere else to go; it is for whatever ends up faster with one.
    auto scaleAttr =
        contraction->getAttrOfType<FloatAttr>("gemmlir.output_scale");
    if (!scaleAttr)
      return failure();
    double scale = scaleAttr.getValueAsDouble();
    if (!(scale > 0.0) || !std::isfinite(scale))
      return failure();

    auto tailTy = llvm::dyn_cast<RankedTensorType>(dcast.getType());
    if (!tailTy || !tailTy.hasStaticShape() || !tailTy.getElementType().isF32())
      return failure();

    // Walk the part of the tail the call can do, then keep going through the
    // relayouts that mean nothing to it. The requantization goes at the end of
    // the *first* part, because the call writes in its own layout.
    Value last = dcast.getResult();
    Value here = last;
    Operation *blocker = nullptr;
    bool joined = false;
    SmallVector<Value> branches;
    for (unsigned step = 0; step < 12; step++) {
      if (!here.hasOneUse()) {
        // A SiLU is `x * sigmoid(x)` and a layer norm is `(x - mean(x)) / ...`,
        // so the value either of them blocks is read **twice**. Stopping at the
        // first branch is how sixteen depthwise convolutions kept their loops
        // after everything else was in place.
        for (Operation *u : here.getUsers())
          if (stopsTheTail(u) || joinsTwoPathsFrom(u, here)) {
            blocker = u;
            joined = !stopsTheTail(u);
            break;
          }
        // A fan-out of slices: an LSTM's gate matrix is one contraction chunked
        // four ways, and each quarter ends in a sigmoid or a tanh, so no single
        // user blocks anything. The cut then goes on **each branch**, below the
        // fan-out, so every gate gets an i8 of its own and the sum above them
        // stays in f32 -- see the measurement below for why not above.
        if (!blocker && llvm::all_of(here.getUsers(), [](Operation *u) {
              return llvm::isa<tensor::ExtractSliceOp, tensor::CollapseShapeOp,
                               tensor::ExpandShapeOp, linalg::TransposeOp>(u);
            })) {
          SmallVector<Operation *> found;
          for (Operation *u : here.getUsers())
            if (Operation *b = blockerBelow(u->getResult(0), 8))
              found.push_back(b);
            else
              break;
          if (found.size() == (size_t)std::distance(here.getUsers().begin(),
                                                    here.getUsers().end())) {
            blocker = found.front();
            for (Operation *u : here.getUsers())
              branches.push_back(u->getResult(0));
          }
        }
        // Tried and reverted, with the measurement. An LSTM's gate matrix is
        // chunked four ways and each quarter ends in a sigmoid or a tanh, so no
        // single user blocks anything and the gates stay in f32 -- **19.9 ms of
        // that model's 37.6**. Following each branch of a fan-out of slices and
        // cutting when they all want it does reach them, and it does not pay:
        //
        //   * cutting *above* the fan-out turns the gate sum into a
        //     `resadd_i8`, which on 192 elements buys 3% (37.6 -> 36.4 ms) and
        //     costs **73% more error**, 0.0038 -> 0.0066 relative L2;
        //   * cutting *below* it does not reach the tables either, because
        //     the three gates fuse into **one** generic that reads three f32
        //     slices of a dequantized buffer, and
        //     `--table-for-i8-elementwise` anchors on a single-input chain
        //     below an i8, which that is not.
        //
        // The 19.9 ms is real and still there; what has to move first is the
        // shape the gates arrive in, not where the cut goes.
        break;
      }
      Operation *user = *here.getUsers().begin();
      if (absorbsIntoTheCall(user, here)) {
        here = user->getResult(0);
        last = here;
        continue;
      }
      if (auto generic = llvm::dyn_cast<linalg::GenericOp>(user)) {
        if (genericAbsorbsIntoTheCall(generic, here)) {
          here = generic.getResult(0);
          last = here;
          continue;
        }
      }
      if (llvm::isa<linalg::TransposeOp, tensor::PadOp, tensor::ExtractSliceOp,
                    tensor::CollapseShapeOp, tensor::ExpandShapeOp>(user)) {
        here = user->getResult(0);
        continue;
      }
      // A sum of two contractions -- an LSTM's `W_ih x + W_hh h`, a ResNet
      // stage transition's projection beside its convolution -- is not a
      // contraction, so the calibration has no range for it. The two ranges
      // added bound it and cannot clip, and measured over an LSTM's 32 gate
      // sums that bound is 1.20x the true range at the median, 1.39x at worst:
      // 0.27 bits.
      if (std::optional<double> s = otherAccumulatorAdded(user, here)) {
        scale += *s;
        here = user->getResult(0);
        last = here;
        continue;
      }
      blocker = user;
      break;
    }
    // Everything that ends a tail and has a pass of its own is left alone --
    // putting a requantization in front of those takes work *away* from the
    // accelerator, and it cost EfficientNet all nine of its `resadd_i8` the
    // first time this fired on anything that merely stopped the walk.
    // Either reason will do: an operation the pipeline cannot fold, or a join of
    // two paths off the accumulator, which no `mvout` can produce. What this
    // still refuses is a blocker that merely *stopped* the walk -- a residual
    // add, a pool, a concatenation, another contraction -- because those have a
    // pass of their own and cutting in front of them takes work away from the
    // accelerator. They are safe from the join rule on their own: their second
    // operand comes from somewhere else, so only one of the two descends from
    // the accumulator.
    if (!blocker || !(stopsTheTail(blocker) || joined))
      return failure();
    SmallVector<Value> cuts = branches.empty() ? SmallVector<Value>{last}
                                               : branches;
    for (Value v : cuts)
      if (v.use_empty() || !llvm::isa<RankedTensorType>(v.getType()))
        return failure();

    MLIRContext *ctx = rewriter.getContext();
    Location loc = dcast.getLoc();
    auto storage = IntegerType::get(ctx, 8);
    auto expressed = rewriter.getF32Type();
    auto qElem = quant::UniformQuantizedType::get(
        quant::QuantizationFlags::Signed, storage, expressed, scale,
        /*zeroPoint=*/0, /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
    OpBuilder::InsertionGuard guard(rewriter);
    for (Value cut : cuts) {
      auto cutTy = llvm::cast<RankedTensorType>(cut.getType());
      rewriter.setInsertionPointAfterValue(cut);
      Value q = rewriter.create<quant::QuantizeCastOp>(loc, cutTy.clone(qElem),
                                                       cut);
      Value storageValue = rewriter.create<quant::StorageCastOp>(
          loc, cutTy.clone(storage), q);
      // Written as arithmetic rather than a `quant.dcast`, because the quant
      // dialect folds `dcast(qcast(x))` back to `x` -- it takes a quantization
      // to be exact -- and the round trip would be erased as fast as it is made.
      Value wide = rewriter.create<arith::SIToFPOp>(loc, cutTy, storageValue);
      Value scaleSplat = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(cutTy, APFloat(static_cast<float>(scale))));
      Value back = rewriter.create<arith::MulFOp>(loc, wide, scaleSplat);
      rewriter.replaceAllUsesExcept(cut, back, q.getDefiningOp());
    }
    return success();
  }
};

class QuantizeUnfoldableTails
    : public impl::QuantizeUnfoldableTailsBase<QuantizeUnfoldableTails> {
public:
  using impl::QuantizeUnfoldableTailsBase<
      QuantizeUnfoldableTails>::QuantizeUnfoldableTailsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, quant::QuantDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<QuantizeBeforeUnfoldableActivation>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
