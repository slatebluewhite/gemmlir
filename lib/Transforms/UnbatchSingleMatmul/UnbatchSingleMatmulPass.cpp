//===- UnbatchSingleMatmulPass.cpp -----------------------------*- C++ -*-===//
//
// A batch of one is not a batch.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_UNBATCHSINGLEMATMUL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// A `linalg.batch_matmul` over one batch is a `linalg.matmul` wearing an extra
/// dimension. Taking the dimension off puts it back on the path where the
/// requantization, the bias and the activation fuse.
class Unbatch : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp batch,
                                PatternRewriter &rewriter) const final {
    if (batch.getInputs().size() != 2 || batch.getOutputs().size() != 1 ||
        batch->getNumResults() != 1)
      return failure();

    SmallVector<RankedTensorType> types;
    for (Value v : {batch.getInputs()[0], batch.getInputs()[1],
                    batch.getOutputs()[0]}) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t || t.getRank() != 3 || !t.hasStaticShape() || t.getDimSize(0) != 1)
        return failure();
      types.push_back(t);
    }
    auto outTy = cast<RankedTensorType>(batch->getResult(0).getType());
    if (outTy.getRank() != 3 || !outTy.hasStaticShape() ||
        outTy.getDimSize(0) != 1)
      return failure();

    Location loc = batch.getLoc();
    SmallVector<ReassociationIndices> merge = {{0, 1}, {2}};

    /// A weight reaches a batched matmul broadcast into the batch: torch-mlir
    /// writes `linalg.generic` copying a 2-D constant into a 1 x M x N one.
    /// Collapsing that back would leave a reshape in front of the constant, and
    /// the folder that turns a constant weight into an i8 one at compile time
    /// walks through permuting copies, not reshapes -- so the weight would be
    /// quantized again on every inference. Three of those is 3072 elements of
    /// an attention head's 6944. Reading through the broadcast instead leaves
    /// the constant where the folder can still see it.
    auto unbroadcast = [&](Value v) -> Value {
      auto generic = v.getDefiningOp<linalg::GenericOp>();
      if (!generic || generic.getInputs().size() != 1 ||
          generic.getOutputs().size() != 1)
        return nullptr;
      Block &body = generic.getRegion().front();
      auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
      if (!yield || yield.getNumOperands() != 1 ||
          yield.getOperand(0) != body.getArgument(0))
        return nullptr;
      Value src = generic.getInputs()[0];
      auto srcTy = dyn_cast<RankedTensorType>(src.getType());
      if (!srcTy || srcTy.getRank() != 2)
        return nullptr;
      // out[b, i, j] = in[i, j], with b the batch this operation has one of.
      MLIRContext *ctx = batch.getContext();
      AffineExpr b, i, j;
      bindDims(ctx, b, i, j);
      SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
      if (maps.size() != 2 ||
          maps[0] != AffineMap::get(3, 0, {i, j}, ctx) ||
          maps[1] != AffineMap::get(3, 0, {b, i, j}, ctx))
        return nullptr;
      return src;
    };

    auto flatten = [&](Value v, RankedTensorType t) -> Value {
      if (Value src = unbroadcast(v))
        return src;
      auto flat = RankedTensorType::get({t.getDimSize(1), t.getDimSize(2)},
                                        t.getElementType());
      // One value collapsed once. Three projections read the same input, and a
      // reshape each would make them three different SSA values -- so the
      // quantization that follows could not be shared either, and the same
      // tensor was quantized three times, 1024 elements of an attention head's
      // 3872.
      //
      // Put it where the value is defined rather than where this matmul is:
      // the greedy driver rewrites the matmuls in the order it finds them, so
      // a reshape placed at one of them does not dominate the others.
      for (Operation *user : v.getUsers())
        if (auto existing = llvm::dyn_cast<tensor::CollapseShapeOp>(user))
          if (existing.getType() == flat &&
              existing.getReassociationIndices() ==
                  SmallVector<ReassociationIndices>(merge))
            return existing.getResult();
      OpBuilder::InsertionGuard guard(rewriter);
      if (Operation *def = v.getDefiningOp())
        rewriter.setInsertionPointAfter(def);
      else
        rewriter.setInsertionPointToStart(v.getParentBlock());
      return rewriter.create<tensor::CollapseShapeOp>(loc, flat, v, merge)
          .getResult();
    };

    Value lhs = flatten(batch.getInputs()[0], types[0]);
    Value rhs = flatten(batch.getInputs()[1], types[1]);
    Value init = flatten(batch.getOutputs()[0], types[2]);
    auto flatOut = RankedTensorType::get(
        {outTy.getDimSize(1), outTy.getDimSize(2)}, outTy.getElementType());

    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, TypeRange{flatOut}, ValueRange{lhs, rhs}, ValueRange{init});
    // The calibration's annotations live on the operation, and the contraction
    // they describe is the same one.
    for (NamedAttribute attr : batch->getDiscardableAttrs())
      matmul->setAttr(attr.getName(), attr.getValue());

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        batch, outTy, matmul.getResult(0), merge);
    return success();
  }
};

class UnbatchSingleMatmul
    : public impl::UnbatchSingleMatmulBase<UnbatchSingleMatmul> {
public:
  using impl::UnbatchSingleMatmulBase<
      UnbatchSingleMatmul>::UnbatchSingleMatmulBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<Unbatch>(&getContext());
    // And the same dimension everywhere else. A transformer's whole graph is
    // `1 x sequence x features`, and taking the batch off the matmul alone
    // leaves it wrapped in reshapes that keep its dequantization and the next
    // layer's quantization on opposite sides -- so neither fuses. linalg knows
    // how to drop a unit extent; this is the one place that wants it.
    linalg::ControlDropUnitDims options;
    // Only the leading one, and only where the shapes are a sequence and its
    // features. A convolution's NHWC tensors have a leading batch of one too,
    // and taking it off leaves the 4-D form every convolution matcher looks
    // for -- measured, twelve of the models stopped compiling and six more lost
    // half their accelerator calls.
    options.controlFn = [](Operation *op) -> SmallVector<unsigned> {
      auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
      if (!generic || generic.getNumLoops() != 3)
        return {};
      for (Value v : generic->getOperands()) {
        auto t = dyn_cast<ShapedType>(v.getType());
        if (!t || t.getRank() != 3 || t.getDimSize(0) != 1)
          return {};
      }
      return {0u};
    };
    linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
