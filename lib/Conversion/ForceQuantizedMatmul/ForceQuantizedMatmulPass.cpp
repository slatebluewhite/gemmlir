//===- ForceQuantizedMatmulPass.cpp -----------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include <cmath>
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <limits>

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FORCEQUANTIZEDMATMUL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The largest magnitude inside one static window of a dense constant.
///
/// `--split-grouped-conv` gives every group a slice of one filter, so the
/// weights reach the matmul behind a `tensor.extract_slice`. Reading the whole
/// constant instead would only cost resolution; not reading it at all is what
/// used to happen, and the caller then fell back to the activation's scale --
/// which for a 3x3 filter is about three of the 256 levels, and came back as a
/// ResNeXt block at 0.1295 relative L2.
static std::optional<double> sliceAbsMax(DenseElementsAttr dense,
                                         ArrayRef<int64_t> offsets,
                                         ArrayRef<int64_t> sizes) {
  ArrayRef<int64_t> shape = llvm::cast<ShapedType>(dense.getType()).getShape();
  unsigned rank = shape.size();
  if (offsets.size() != rank || sizes.size() != rank)
    return std::nullopt;
  SmallVector<double> values;
  for (const llvm::APFloat &f : dense.getValues<llvm::APFloat>())
    values.push_back(static_cast<double>(f.convertToFloat()));

  SmallVector<int64_t> strides(rank, 1);
  for (int d = static_cast<int>(rank) - 2; d >= 0; d--)
    strides[d] = strides[d + 1] * shape[d + 1];

  int64_t count = 1;
  for (unsigned d = 0; d < rank; d++) {
    if (offsets[d] < 0 || sizes[d] < 0 || offsets[d] + sizes[d] > shape[d])
      return std::nullopt;
    count *= sizes[d];
  }

  double m = 0.0;
  SmallVector<int64_t> index(rank, 0);
  for (int64_t n = 0; n < count; n++) {
    int64_t flat = 0;
    for (unsigned d = 0; d < rank; d++)
      flat += (offsets[d] + index[d]) * strides[d];
    if (flat < 0 || flat >= static_cast<int64_t>(values.size()))
      return std::nullopt;
    m = std::max(m, std::fabs(values[flat]));
    for (int d = static_cast<int>(rank) - 1; d >= 0; d--) {
      if (++index[d] < sizes[d])
        break;
      index[d] = 0;
    }
  }
  return m;
}

/// Largest magnitude in a constant operand, looking through pure copies.
///
/// A frontend rarely hands the matmul a bare constant: torch-mlir, for one,
/// transposes the weights with a `linalg.generic` that only yields its input.
/// Anything that computes something is not followed -- its range is not the
/// constant's.
static std::optional<double> constantAbsMax(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto cst = llvm::dyn_cast<arith::ConstantOp>(def)) {
      auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense || !dense.getElementType().isF32())
        return std::nullopt;
      double m = 0.0;
      for (const llvm::APFloat &f : dense.getValues<llvm::APFloat>())
        m = std::max(m, std::fabs(static_cast<double>(f.convertToFloat())));
      return m;
    }
    // A reshape does not touch the values, so the largest is still the
    // largest. `--unbatch-single-matmul` puts one in front of every weight it
    // takes a batch dimension off, and without this the weight falls back to
    // the activation's scale -- one attention head went from 0.0115 to 0.0643
    // relative L2 on exactly that.
    if (auto collapse = llvm::dyn_cast<tensor::CollapseShapeOp>(def)) {
      v = collapse.getSrc();
      continue;
    }
    if (auto expand = llvm::dyn_cast<tensor::ExpandShapeOp>(def)) {
      v = expand.getSrc();
      continue;
    }
    if (auto slice = llvm::dyn_cast<tensor::ExtractSliceOp>(def)) {
      if (!slice.hasUnitStride())
        return std::nullopt;
      ArrayRef<int64_t> offsets = slice.getStaticOffsets();
      ArrayRef<int64_t> sizes = slice.getStaticSizes();
      if (llvm::any_of(offsets, ShapedType::isDynamic) ||
          llvm::any_of(sizes, ShapedType::isDynamic))
        return std::nullopt;
      auto cst = slice.getSource().getDefiningOp<arith::ConstantOp>();
      if (!cst)
        return std::nullopt;
      auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense || !dense.getElementType().isF32())
        return std::nullopt;
      return sliceAbsMax(dense, offsets, sizes);
    }
    auto generic = llvm::dyn_cast<linalg::GenericOp>(def);
    if (!generic || generic.getInputs().size() != 1 ||
        generic.getOutputs().size() != 1)
      return std::nullopt;
    Block &body = generic.getRegion().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != body.getArgument(0))
      return std::nullopt;
    v = generic.getInputs()[0];
  }
  return std::nullopt;
}

/// Walks to a constant through pure copies, composing the permutation on the
/// way, so the constant can be read in the operand's own index order.
///
/// Returns the constant and a map from the operand's loop dimensions to the
/// constant's. Only identity output maps and permuted inputs are followed --
/// a transpose is the case that matters, since that is how a frontend hands
/// weights over.
static std::optional<std::pair<DenseElementsAttr, AffineMap>>
constantThroughCopies(Value v) {
  MLIRContext *ctx = v.getContext();
  auto rank = llvm::cast<RankedTensorType>(v.getType()).getRank();
  AffineMap composed = AffineMap::getMultiDimIdentityMap(rank, ctx);

  while (Operation *def = v.getDefiningOp()) {
    if (auto cst = llvm::dyn_cast<arith::ConstantOp>(def)) {
      auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense || !dense.getElementType().isF32())
        return std::nullopt;
      return std::make_pair(dense, composed);
    }
    // A reshape that only drops leading unit axes. torch-mlir writes a batched
    // matmul's weight broadcast at rank 4 -- `1 x B x K x N` -- and collapses
    // the leading pair away before the contraction reads it, so without this
    // the walk stops one step short of the broadcast and ConvNeXt's nine MLP
    // weights are converted on every inference. Only unit axes, because the
    // running map is per-dimension and a real merge would change what each
    // index means.
    if (auto collapse = llvm::dyn_cast<tensor::CollapseShapeOp>(def)) {
      auto inTy = llvm::cast<RankedTensorType>(collapse.getSrc().getType());
      auto outTy = llvm::cast<RankedTensorType>(collapse.getType());
      if (!inTy.hasStaticShape() || inTy.getNumElements() != outTy.getNumElements())
        return std::nullopt;
      SmallVector<int64_t> kept;
      for (int64_t d = 0; d < inTy.getRank(); d++)
        if (inTy.getDimSize(d) != 1)
          kept.push_back(d);
      if (kept.size() != (size_t)outTy.getRank())
        return std::nullopt;
      // Re-read the running map at the source's rank: result r of the old map
      // named output dimension r, which is input dimension kept[r].
      SmallVector<AffineExpr> results(inTy.getRank(),
                                      getAffineConstantExpr(0, ctx));
      for (auto [r, d] : llvm::enumerate(kept))
        results[d] = composed.getResult(r);
      composed = AffineMap::get(composed.getNumDims(), 0, results, ctx);
      v = collapse.getSrc();
      continue;
    }
    auto generic = llvm::dyn_cast<linalg::GenericOp>(def);
    if (!generic || generic.getInputs().size() != 1 ||
        generic.getOutputs().size() != 1)
      return std::nullopt;
    Block &body = generic.getRegion().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != body.getArgument(0))
      return std::nullopt;
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    // The write has to be a permutation -- every destination element written
    // once. The **read** need not be: a broadcast reads through a map that
    // drops a dimension, and that is how a weight reaches a
    // `linalg.batch_matmul`, which takes a weight of the same rank as the
    // activation. Refusing it left ConvNeXt converting a 768 x 3072 weight on
    // every inference. The evaluation below already unravels the destination
    // index and maps it into the source, so a dropped dimension needs nothing
    // extra there; what it needs is for the *result* to stay small, which is
    // `quantizedConstant`'s job.
    if (maps.size() != 2 || !maps[1].isPermutation())
      return std::nullopt;
    if (!llvm::all_of(maps[0].getResults(), [](AffineExpr e) {
          return llvm::isa<AffineDimExpr>(e);
        }))
      return std::nullopt;
    // out[maps[1](i)] = in[maps[0](i)], so out index j reads
    // in[maps[0](inverse(maps[1])(j))]. A frontend transposes by permuting the
    // *output* map, which is why both sides have to be handled.
    AffineMap inverseOut = inversePermutation(maps[1]);
    if (!inverseOut)
      return std::nullopt;
    composed = maps[0].compose(inverseOut).compose(composed);
    v = generic.getInputs()[0];
    // The walk continues at the source's own rank, so the running map has to
    // be read at that rank from here on.
    if (composed.getNumResults() !=
        (unsigned)llvm::cast<RankedTensorType>(v.getType()).getRank())
      return std::nullopt;
  }
  return std::nullopt;
}

/// One f32 constant read through `map`, quantized element for element.
///
/// Exactly what the runtime path computes: `--lower-quant-ops` divides in f32
/// and `--round-quantized-casts` rounds half to even, then clamps. Staying in
/// float keeps the folded constant bit-identical to the loop it replaces.
static std::optional<DenseElementsAttr>
foldQuantized(DenseElementsAttr dense, AffineMap map, double scale,
              RankedTensorType resultTy) {
  if (scale == 0.0 || !dense.getElementType().isF32())
    return std::nullopt;
  ArrayRef<int64_t> shape = resultTy.getShape();
  auto srcTy = llvm::cast<ShapedType>(dense.getType());
  ArrayRef<int64_t> srcShape = srcTy.getShape();
  if ((int64_t)shape.size() != map.getNumDims() ||
      (int64_t)srcShape.size() != map.getNumResults())
    return std::nullopt;

  SmallVector<float> values;
  values.reserve(dense.getNumElements());
  for (const llvm::APFloat &f : dense.getValues<llvm::APFloat>())
    values.push_back(f.convertToFloat());

  int64_t total = resultTy.getNumElements();
  SmallVector<llvm::APInt> out;
  out.reserve(total);
  SmallVector<int64_t> idx(shape.size(), 0);
  for (int64_t linear = 0; linear < total; linear++) {
    int64_t rem = linear;
    for (int64_t d = shape.size() - 1; d >= 0; d--) {
      idx[d] = rem % shape[d];
      rem /= shape[d];
    }
    int64_t srcLinear = 0;
    for (unsigned r = 0; r < map.getNumResults(); r++) {
      auto dim = llvm::dyn_cast<AffineDimExpr>(map.getResult(r));
      if (!dim)
        return std::nullopt;
      srcLinear = srcLinear * srcShape[r] + idx[dim.getPosition()];
    }
    float scaled = values[srcLinear] / static_cast<float>(scale);
    float r = std::nearbyint(scaled);
    int64_t q = (int64_t)std::max(-128.0f, std::min(127.0f, r));
    out.push_back(llvm::APInt(8, q, /*isSigned=*/true));
  }
  return DenseElementsAttr::get(resultTy, out);
}

/// When `v` is a constant broadcast up to `wideTy`, the constant's own type and
/// the map that spreads it.
///
/// `linalg.batch_matmul` takes a weight of the same rank as the activation, so
/// a frontend copies the 2-D constant once per batch element. Folding at the
/// wide shape is correct but doubles the constant in the binary for a batch of
/// two; folding at the narrow one and broadcasting i8 is the same bytes on the
/// wire and a `memcpy` at run time.
/// The operand, already quantized, as an i8 constant -- so the conversion is
/// not repeated on every inference.
///
/// A frontend does not hand the matmul a bare constant: torch-mlir transposes
/// the weights first, and `--force-quantized-matmul` then puts a conversion on
/// top. MLIR's own linalg constant folder will not collapse that chain, because
/// it requires every operand to share an element type and this one is f32 in,
/// i8 out. Measured on a small CNN: 4928 elements re-converted per run.
static std::optional<DenseElementsAttr>
quantizedConstant(Value v, double scale, RankedTensorType resultTy) {
  std::optional<std::pair<DenseElementsAttr, AffineMap>> found =
      constantThroughCopies(v);
  if (!found)
    return std::nullopt;
  return foldQuantized(found->first, found->second, scale, resultTy);
}

/// Symmetric int8 scale for an operand: derived when it is constant, otherwise
/// the caller-supplied activation scale.
static double operandScale(Value v, double fallback) {
  std::optional<double> absMax = constantAbsMax(v);
  if (!absMax || *absMax == 0.0)
    return fallback;
  return *absMax / 127.0;
}

/// Per-operation activation scale, if a calibration step annotated it.
///
/// One scale cannot serve a whole network: on a two-layer MLP the inputs to the
/// two matmuls reached 3.74 and 2.11, so whichever scale is chosen wastes range
/// in one of them. `gemmlir.activation_scale` lets whatever measured the ranges
/// say so per operation; the pass option remains the fallback.
/// Taking the fallback is said out loud. One layer quantized at a scale nobody
/// measured used to be invisible: `cnn_i2c` reached the board with its first
/// convolution running in f32 because the contraction arrived as a
/// `linalg.generic` and calibration never saw it, and the only way to notice
/// was to count the loops in the final IR. A warning here is the difference
/// between a number that was chosen and a number that was left.
static double activationScaleOf(Operation *op, double fallback) {
  if (auto attr = op->getAttrOfType<FloatAttr>("gemmlir.activation_scale")) {
    double v = attr.getValueAsDouble();
    if (v > 0.0)
      return v;
  }
  op->emitWarning() << "no gemmlir.activation_scale; quantizing this "
                    << "contraction at the fallback " << fallback
                    << ". Calibrate it -- and if a frontend packed im2col, "
                    << "run --raise-contraction-to-matmul before calibrating, "
                    << "so the annotation lands on an operation calibrate.py "
                    << "can see.";
  return fallback;
}

/// The right-hand operand's scale, when it is not a constant.
///
/// A convolution or a linear layer multiplies an activation by a weight, and
/// the weight's range is there to be read. A transformer's two busiest
/// contractions do not: `Q @ K.T` and `probs @ V` are activation times
/// activation, and the right-hand range is nobody's to guess. Falling back to
/// the left-hand one is what used to happen and it is not close -- attention
/// probabilities live in [0, 0.13] and the values they weigh reach 2.8, so
/// quantizing the values at the probabilities' scale saturates them flat.
/// Measured on one head, 0.8352 relative L2.
///
/// `gemmlir.rhs_activation_scale` is the calibrated answer. Without it there is
/// no honest number, and the contraction is left alone.
static std::optional<double> rhsActivationScaleOf(Operation *op) {
  if (auto attr = op->getAttrOfType<FloatAttr>("gemmlir.rhs_activation_scale")) {
    double v = attr.getValueAsDouble();
    if (v > 0.0)
      return v;
  }
  return std::nullopt;
}

  // Rewrites a tensor linalg contraction (f32,f32)->f32 into a quantized
  // pipeline:
  // qcast(A,f32->!quant.uniform<i8:f32,0.02>) -> scast -> i8
  // qcast(B,f32->!quant.uniform<i8:f32,0.02>) -> scast -> i8
  // C = fill 0 : tensor<...xi32>
  // op(i8,i8)->i32
  // scast(i32 -> !quant.uniform<i32:f32, 0.02*0.02>) -> dcast -> f32
  //
  // Nothing here is specific to matmul beyond "two inputs, one output, and the
  // named op extends its i8 operands into the i32 accumulator itself", which is
  // why it is also instantiated for `linalg.conv_2d_nhwc_hwcf`: that is the form
  // --convert-linalg-to-gemmlir already knows how to fold into `conv2d_i8` with
  // the bias, requantization, relu and pooling attached.
  template <typename OpTy>
  class LinalgOpQuantizer : public OpRewritePattern<OpTy> {
    double activationScale;
  public:
    LinalgOpQuantizer(MLIRContext *ctx, double activationScale)
        : OpRewritePattern<OpTy>(ctx), activationScale(activationScale) {}

    LogicalResult matchAndRewrite(OpTy op,
                                  PatternRewriter &rewriter) const final {
      Location loc = op.getLoc();

      // Expect exactly 2 inputs and 1 output (tensor form).
      if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
        return rewriter.notifyMatchFailure(op, "expected 2 inputs and 1 output");

      Value lhs = op.getInputs()[0];
      Value rhs = op.getInputs()[1];
      Value outInit = op.getOutputs()[0];

      auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
      auto outTy = dyn_cast<RankedTensorType>(outInit.getType());

      if (!lhsTy || !rhsTy || !outTy)
        return rewriter.notifyMatchFailure(op, "only supports tensor operands/results");

      if (!lhsTy.getElementType().isF32() || !rhsTy.getElementType().isF32() ||
          !outTy.getElementType().isF32())
        return rewriter.notifyMatchFailure(op, "expects f32 element types");

      // Static shapes for simplicity (can be extended to dynamic later).
      if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() || !outTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "requires static shapes");

      // Construct per-tensor symmetric uniform quant types.
      auto f32 = rewriter.getF32Type();
      auto i8 = rewriter.getIntegerType(8);
      auto i32 = rewriter.getI32Type();

      // A symmetric int8 scale is max|x|/127. For a constant operand -- the
      // weights -- that is known here; for an activation it is not, and a
      // representative value has to be supplied instead. Quantizing everything
      // at one fixed scale, as this pass used to, saturates or flattens most of
      // a normally-distributed tensor: on a PyTorch MLP it cost a relative L2
      // error of 0.50.
      double actScale = activationScaleOf(op, activationScale);
      double scaleLhs = operandScale(lhs, actScale);
      // The constant's own range first -- it is exact where a calibration is a
      // sample -- then what was measured for it, then the left operand's, which
      // is only ever a guess. See `rhsActivationScaleOf` for what that guess
      // costs when the right operand is an activation.
      double scaleRhs;
      if (std::optional<double> constant = constantAbsMax(rhs))
        scaleRhs = *constant == 0.0 ? actScale : *constant / 127.0;
      else if (std::optional<double> measured = rhsActivationScaleOf(op))
        scaleRhs = *measured;
      else
        scaleRhs = actScale;
      double scaleAcc = scaleLhs * scaleRhs;

      auto qI8Of = [&](double scale) {
        return quant::UniformQuantizedType::get(
            quant::QuantizationFlags::Signed, /*storageType=*/i8,
            /*expressedType=*/f32, /*scale=*/scale, /*zeroPoint=*/0,
            /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
      };
      auto qI8Lhs = qI8Of(scaleLhs);
      auto qI8Rhs = qI8Of(scaleRhs);

      auto qI32 = quant::UniformQuantizedType::get(
          quant::QuantizationFlags::Signed, /*storageType=*/i32,
          /*expressedType=*/f32, /*scale=*/scaleAcc, /*zeroPoint=*/0,
          /*storageTypeMin=*/std::numeric_limits<int32_t>::min(),
          /*storageTypeMax=*/std::numeric_limits<int32_t>::max());

      auto lhsQTy = RankedTensorType::get(lhsTy.getShape(), qI8Lhs);
      auto rhsQTy = RankedTensorType::get(rhsTy.getShape(), qI8Rhs);
      auto lhsI8Ty = RankedTensorType::get(lhsTy.getShape(), i8);
      auto rhsI8Ty = RankedTensorType::get(rhsTy.getShape(), i8);
      auto outI32Ty = RankedTensorType::get(outTy.getShape(), i32);
      auto outQI32Ty = RankedTensorType::get(outTy.getShape(), qI32);

      // A constant operand is converted here rather than on every inference.
      auto quantizeOperand = [&](Value v, double scale, RankedTensorType qTy,
                                 RankedTensorType i8Ty) -> Value {
        if (std::optional<DenseElementsAttr> folded =
                quantizedConstant(v, scale, i8Ty))
          return rewriter.create<arith::ConstantOp>(loc, i8Ty, *folded);
        // One value quantized at one scale is one loop, however many
        // contractions read it. A transformer's three projections all read the
        // same input at the same scale, and quantizing it once each is 1024
        // elements of an attention head's 3872. Plain CSE would merge these and
        // also merge the accumulators' zero fills, which then have more than
        // one user and stop proving that the accumulator starts at zero -- so
        // the requantizations stop folding and the net is worse. This merges
        // the casts and nothing else.
        for (Operation *user : v.getUsers()) {
          auto existing = llvm::dyn_cast<quant::QuantizeCastOp>(user);
          if (!existing || existing.getType() != qTy)
            continue;
          for (Operation *second : existing->getUsers())
            if (auto storage = llvm::dyn_cast<quant::StorageCastOp>(second))
              if (storage.getType() == i8Ty)
                return storage.getResult();
        }
        // At the value's definition, not at this contraction: the greedy
        // driver rewrites the contractions in the order it finds them, so a
        // cast placed at one of them does not dominate the others and the
        // sharing above could never fire.
        OpBuilder::InsertionGuard guard(rewriter);
        if (Operation *def = v.getDefiningOp())
          rewriter.setInsertionPointAfter(def);
        else
          rewriter.setInsertionPointToStart(v.getParentBlock());
        Value q = rewriter.create<quant::QuantizeCastOp>(loc, qTy, v);
        return rewriter.create<quant::StorageCastOp>(loc, i8Ty, q);
      };
      Value lhsI8 = quantizeOperand(lhs, scaleLhs, lhsQTy, lhsI8Ty);
      Value rhsI8 = quantizeOperand(rhs, scaleRhs, rhsQTy, rhsI8Ty);

      // linalg.matmul accumulates into its output operand, and a frontend puts
      // things there: torch-mlir broadcasts a convolution's bias into the init
      // tensor rather than adding it afterwards. Starting from zero would drop
      // it silently -- on a PyTorch CNN that cost a relative L2 error of 0.33
      // where 0.013 was available. The incoming values are f32, so they are
      // added back after the dequantization rather than quantized.
      bool zeroInit = false;
      if (auto fill = outInit.getDefiningOp<linalg::FillOp>())
        if (fill.getInputs().size() == 1 &&
            matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
          zeroInit = true;

      // C_base: tensor.empty : tensor<...xi32>
      Value c0 = rewriter.create<arith::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(0));
      // Create an empty tensor with the desired i32 result type. In MLIR 22,
      // tensor::EmptyOp builders take either sizes+element type or the result
      // type with dynamic sizes. All static dims -> no dynamic sizes.
      Value cbase = tensor::EmptyOp::create(rewriter, loc, outI32Ty, ValueRange{}).getResult();
      // C = linalg.fill ins(%c0) outs(%cbase) -> tensor<...xi32>
      Value cFilled = rewriter.create<linalg::FillOp>(loc, ValueRange{c0}, ValueRange{cbase})
                             .getResult(0);

      // result_i32 = <same op> ins(lhsI8, rhsI8) outs(cFilled) -> tensor<...xi32>
      auto quantOp = rewriter.create<OpTy>(loc, TypeRange{outI32Ty},
                                           ValueRange{lhsI8, rhsI8},
                                           ValueRange{cFilled});
      // What the operation *means* lives in these: `indexing_maps` is how MLIR
      // 22 says a matmul operand is transposed, and `strides`/`dilations` are
      // the convolution's window. Dropping any of them quietly computes
      // something else, so they are carried over rather than defaulted.
      for (StringRef name : {"indexing_maps", "strides", "dilations"})
        if (Attribute a = op->getAttr(name))
          quantOp->setAttr(name, a);
      // And what the *calibration* wrote about this layer's own output.
      // `--quantize-unfoldable-tails` has nothing else to put a requantization
      // at, and dropping it here left EfficientNet's sixteen depthwise
      // convolutions on the core with nothing saying why. The input scales are
      // not carried: they have been read by this point and re-emitting them
      // only changes the IR's surface. Same lesson as
      // --raise-contraction-to-matmul -- when a rewrite replaces an operation,
      // ask what was written on the old one.
      if (Attribute a = op->getAttr("gemmlir.output_scale"))
        quantOp->setAttr("gemmlir.output_scale", a);
      Value resultI32 = quantOp.getResult(0);

      // result_quant = quant.scast i32 -> quantized<i32:f32, 0.0004>
      Value resultQ = rewriter.create<quant::StorageCastOp>(loc, outQI32Ty, resultI32);
      // result = quant.dcast -> f32
      auto outF32Ty = RankedTensorType::get(outTy.getShape(), f32);
      Value resultF32 = rewriter.create<quant::DequantizeCastOp>(loc, outF32Ty, resultQ);
      if (!zeroInit)
        resultF32 = rewriter.create<arith::AddFOp>(loc, resultF32, outInit);

      rewriter.replaceOp(op, resultF32);
      return success();
    }
  };

  // Conversion pass that applies the rewrite greedily.
  class ForceQuantizedMatmul
      : public impl::ForceQuantizedMatmulBase<ForceQuantizedMatmul> {
public:
    using impl::ForceQuantizedMatmulBase<
        ForceQuantizedMatmul>::ForceQuantizedMatmulBase;

    void getDependentDialects(DialectRegistry& registry) const final
    {
      registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                      memref::MemRefDialect, func::FuncDialect,
                      tensor::TensorDialect, quant::QuantDialect>();
    }

    void runOnOperation() final
    {
      ModuleOp module = getOperation();
      RewritePatternSet patterns(&getContext());
      patterns.add<LinalgOpQuantizer<linalg::MatmulOp>,
                   LinalgOpQuantizer<linalg::BatchMatmulOp>,
                   LinalgOpQuantizer<linalg::Conv2DNhwcHwcfOp>,
                   LinalgOpQuantizer<linalg::DepthwiseConv2DNhwcHwcOp>>(
          &getContext(), activationScale);
      if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
        signalPassFailure();
    }
  };

} // namespace

} // namespace mlir::gemmlir
