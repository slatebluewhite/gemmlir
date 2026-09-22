//===- RaiseContractionToMatmulPass.cpp ---------------------*- C++ -*-===//
//
// Turns a contraction whose batch dimensions are all 1 into linalg.matmul.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_RAISECONTRACTIONTOMATMUL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// Collapses every leading unit dimension of `v`, leaving the last two.
static Value dropUnitBatch(PatternRewriter &rewriter, Location loc, Value v,
                           RankedTensorType type) {
  if (type.getRank() == 2)
    return v;
  SmallVector<ReassociationIndices> reassoc;
  ReassociationIndices leading;
  for (int64_t i = 0; i < type.getRank() - 1; i++)
    leading.push_back(i);
  reassoc.push_back(leading);
  reassoc.push_back({type.getRank() - 1});
  auto collapsed = RankedTensorType::get(
      {type.getShape()[type.getRank() - 2], type.getShape()[type.getRank() - 1]},
      type.getElementType());
  return rewriter.create<tensor::CollapseShapeOp>(loc, collapsed, v, reassoc);
}

/// The (row, column) a map assigns to an operand, as contraction dimensions,
/// ignoring dimensions of extent 1. Returns nullopt when the operand is not a
/// plain 2-D view of two of them.
static std::optional<std::pair<unsigned, unsigned>>
operandDims(AffineMap map, ArrayRef<int64_t> shape) {
  SmallVector<unsigned> dims;
  for (auto [i, expr] : llvm::enumerate(map.getResults())) {
    auto d = llvm::dyn_cast<AffineDimExpr>(expr);
    if (!d)
      return std::nullopt;
    if (shape[i] == 1 && map.getNumResults() > 2)
      continue; // a unit batch axis
    dims.push_back(d.getPosition());
  }
  if (dims.size() != 2)
    return std::nullopt;
  return std::make_pair(dims[0], dims[1]);
}

/// A batch-of-one contraction is a matmul with extra axes. Raising it lets
/// everything downstream -- quantization, the gemmlir conversion -- see a named
/// operation, which is what they match on.
class RaiseContraction : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();
    if (!linalg::isaContractionOpInterface(op))
      return failure();

    FailureOr<linalg::ContractionDimensions> dims =
        linalg::inferContractionDims(op);
    if (failed(dims))
      return failure();

    // A unit batch axis that only some operands carry is not classified as a
    // batch dimension -- img2col leaves one that appears in the right-hand side
    // and the result but not the left, so linalg calls it a second `n`. Pick out
    // the dimension of each kind that actually has extent, and require every
    // other one to be 1, since those are the axes being dropped.
    SmallVector<int64_t, 4> loops = op.getStaticLoopRanges();
    auto sole = [&](ArrayRef<unsigned> group) -> std::optional<unsigned> {
      std::optional<unsigned> found;
      for (unsigned d : group) {
        if (loops[d] == 1)
          continue;
        if (found)
          return std::nullopt;
        found = d;
      }
      return found ? found : (group.empty() ? std::nullopt
                                            : std::optional<unsigned>(group[0]));
    };
    std::optional<unsigned> mOpt = sole(dims->m), nOpt = sole(dims->n),
                            kOpt = sole(dims->k);
    if (!mOpt || !nOpt || !kOpt)
      return failure();

    SmallVector<Value> operands = {op.getInputs()[0], op.getInputs()[1],
                                   op.getOutputs()[0]};
    SmallVector<RankedTensorType> types;
    for (Value v : operands) {
      auto t = llvm::dyn_cast<RankedTensorType>(v.getType());
      if (!t || !t.hasStaticShape())
        return failure();
      types.push_back(t);
    }

    unsigned m = *mOpt, n = *nOpt, k = *kOpt;
    // Everything not carrying the matmul has to be an axis of one.
    for (unsigned d = 0; d < loops.size(); d++)
      if (d != m && d != n && d != k && loops[d] != 1)
        return failure();

    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    SmallVector<std::pair<unsigned, unsigned>> flat;
    for (auto [map, type] : llvm::zip_equal(maps, types)) {
      std::optional<std::pair<unsigned, unsigned>> d =
          operandDims(map, type.getShape());
      if (!d)
        return failure();
      flat.push_back(*d);
    }
    if (flat[2] != std::make_pair(m, n))
      return failure();

    MLIRContext *ctx = rewriter.getContext();
    AffineExpr em, en, ek;
    bindDims(ctx, em, en, ek);
    auto mapOf = [&](std::pair<unsigned, unsigned> d) -> AffineMap {
      auto pick = [&](unsigned x) { return x == m ? em : (x == n ? en : ek); };
      return AffineMap::get(3, 0, {pick(d.first), pick(d.second)}, ctx);
    };
    // Only the four orientations linalg.matmul can describe.
    if (!((flat[0] == std::make_pair(m, k) || flat[0] == std::make_pair(k, m)) &&
          (flat[1] == std::make_pair(k, n) || flat[1] == std::make_pair(n, k))))
      return failure();

    Location loc = op.getLoc();
    Value lhs = dropUnitBatch(rewriter, loc, operands[0], types[0]);
    Value rhs = dropUnitBatch(rewriter, loc, operands[1], types[1]);
    Value out = dropUnitBatch(rewriter, loc, operands[2], types[2]);

    SmallVector<Attribute> newMaps = {AffineMapAttr::get(mapOf(flat[0])),
                                      AffineMapAttr::get(mapOf(flat[1])),
                                      AffineMapAttr::get(mapOf({m, n}))};
    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, TypeRange{out.getType()}, ValueRange{lhs, rhs}, ValueRange{out},
        rewriter.getArrayAttr(newMaps));

    // Whatever was written on the contraction belongs on the matmul that
    // replaces it. `calibrate.py` leaves the layer's measured range there as
    // `gemmlir.activation_scale`, and dropping it is silent: the layer simply
    // never gets quantized and runs as an f32 loop nest. That is what happened
    // to `cnn_i2c`'s first convolution -- 16x196x27 multiply-adds in software,
    // in a model whose other layers were all on the accelerator.
    for (NamedAttribute attr : op->getDiscardableAttrs())
      matmul->setAttr(attr.getName(), attr.getValue());

    Value result = matmul.getResult(0);
    if (types[2].getRank() != 2) {
      SmallVector<ReassociationIndices> reassoc;
      ReassociationIndices leading;
      for (int64_t i = 0; i < types[2].getRank() - 1; i++)
        leading.push_back(i);
      reassoc.push_back(leading);
      reassoc.push_back({types[2].getRank() - 1});
      result = rewriter.create<tensor::ExpandShapeOp>(loc, types[2], result,
                                                      reassoc);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class RaiseContractionToMatmul
    : public impl::RaiseContractionToMatmulBase<RaiseContractionToMatmul> {
public:
  using impl::RaiseContractionToMatmulBase<
      RaiseContractionToMatmul>::RaiseContractionToMatmulBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseContraction>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
