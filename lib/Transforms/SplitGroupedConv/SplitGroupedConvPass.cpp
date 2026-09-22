//===- SplitGroupedConvPass.cpp --------------------------------*- C++ -*-===//
//
// A grouped convolution is G convolutions that do not talk to each other.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_SPLITGROUPEDCONV
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// Group `g` of a grouped convolution reads input channels
/// `[g*C/G, (g+1)*C/G)` and writes output channels `[g*F/G, (g+1)*F/G)`.
/// Nothing crosses between groups, so the operation is G ordinary
/// convolutions over channel slices.
///
/// The slices are taken in NHWC, not NCHW. Splitting first and relayouting
/// afterwards gives every group its own pair of transposes -- 64 of them for a
/// ResNeXt block -- where doing it in this order gives one transpose for the
/// whole input and one for the whole result, whatever G is. A channel slice is
/// contiguous in both layouts, and in NHWC it is exactly the strided window
/// `tiled_conv_stride_auto` already takes for a concatenation.
class SplitGrouped : public OpRewritePattern<linalg::Conv2DNgchwGfchwOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNgchwGfchwOp conv,
                                PatternRewriter &rewriter) const final {
    if (conv.getInputs().size() != 2 || conv.getOutputs().size() != 1 ||
        conv->getNumResults() != 1)
      return failure();
    auto inTy = dyn_cast<RankedTensorType>(conv.getInputs()[0].getType());
    auto fTy = dyn_cast<RankedTensorType>(conv.getInputs()[1].getType());
    auto outTy = dyn_cast<RankedTensorType>(conv->getResult(0).getType());
    if (!inTy || !fTy || !outTy || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !outTy.hasStaticShape())
      return failure();
    // (N, G, C/G, H, W), (G, F/G, C/G, KH, KW), (N, G, F/G, OH, OW)
    if (inTy.getRank() != 5 || fTy.getRank() != 5 || outTy.getRank() != 5)
      return failure();

    int64_t batch = inTy.getDimSize(0), groups = inTy.getDimSize(1);
    int64_t inPerGroup = inTy.getDimSize(2);
    int64_t rows = inTy.getDimSize(3), cols = inTy.getDimSize(4);
    int64_t outPerGroup = outTy.getDimSize(2);
    int64_t outRows = outTy.getDimSize(3), outCols = outTy.getDimSize(4);
    int64_t kh = fTy.getDimSize(3), kw = fTy.getDimSize(4);
    if (groups < 2 || fTy.getDimSize(0) != groups ||
        outTy.getDimSize(0) != batch || outTy.getDimSize(1) != groups ||
        fTy.getDimSize(1) != outPerGroup || fTy.getDimSize(2) != inPerGroup)
      return failure();

    Location loc = conv.getLoc();
    Type elem = inTy.getElementType();
    int64_t channels = groups * inPerGroup, filters = groups * outPerGroup;

    auto idx = [&](ArrayRef<int64_t> v) {
      SmallVector<OpFoldResult> r;
      for (int64_t x : v)
        r.push_back(rewriter.getIndexAttr(x));
      return r;
    };
    auto transpose = [&](Value v, ArrayRef<int64_t> shape,
                         ArrayRef<int64_t> perm) {
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, shape, cast<RankedTensorType>(v.getType()).getElementType());
      return rewriter.create<linalg::TransposeOp>(loc, v, init, perm)
          .getResult()[0];
    };

    // (N, G, C/G, H, W) -> (N, C, H, W) -> (N, H, W, C)
    SmallVector<ReassociationIndices> mergeGroups = {{0}, {1, 2}, {3}, {4}};
    Value flatIn = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get({batch, channels, rows, cols}, elem),
        conv.getInputs()[0], mergeGroups);
    Value nhwcIn = transpose(flatIn, {batch, rows, cols, channels}, {0, 2, 3, 1});

    // (G, F/G, C/G, KH, KW) -> (F, C/G, KH, KW)
    SmallVector<ReassociationIndices> mergeFilters = {{0, 1}, {2}, {3}, {4}};
    Value flatFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get({filters, inPerGroup, kh, kw},
                                   fTy.getElementType()),
        conv.getInputs()[1], mergeFilters);

    // The destination the convolution accumulates into, in the same layout.
    Value flatOut = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get({batch, filters, outRows, outCols},
                                   outTy.getElementType()),
        conv.getOutputs()[0], mergeGroups);
    Value join =
        transpose(flatOut, {batch, outRows, outCols, filters}, {0, 2, 3, 1});

    // Transposed whole and sliced after, not the other way round: the filter
    // is a constant, and one transpose of a constant folds away at compile
    // time where G of them, each behind a slice, do not.
    Value hwcfFilter = transpose(flatFilter, {kh, kw, inPerGroup, filters},
                                 {2, 3, 1, 0});

    auto inSliceTy =
        RankedTensorType::get({batch, rows, cols, inPerGroup}, elem);
    auto fSliceTy = RankedTensorType::get({kh, kw, inPerGroup, outPerGroup},
                                          fTy.getElementType());
    auto outSliceTy = RankedTensorType::get(
        {batch, outRows, outCols, outPerGroup}, outTy.getElementType());
    SmallVector<OpFoldResult> unit(4, rewriter.getIndexAttr(1));

    // The groups are joined with a `tensor.concat`, not written one at a time
    // into a shared destination. They are the same thing, but the concatenation
    // is the one the rest of the pipeline already knows: the requantization
    // after it is pushed back into each group, so each group's convolution ends
    // in a requantize and folds, and the join itself becomes the `out_stride`
    // of `tiled_conv_stride_auto`. Written as a chain of `tensor.insert_slice`
    // the groups keep an f32 tail each and none of them folds.
    SmallVector<Value> parts;
    for (int64_t g = 0; g < groups; g++) {
      Value input = rewriter.create<tensor::ExtractSliceOp>(
          loc, inSliceTy, nhwcIn, idx({0, 0, 0, g * inPerGroup}),
          idx({batch, rows, cols, inPerGroup}), unit);
      Value hwcf = rewriter.create<tensor::ExtractSliceOp>(
          loc, fSliceTy, hwcfFilter, idx({0, 0, 0, g * outPerGroup}),
          idx({kh, kw, inPerGroup, outPerGroup}), unit);
      Value init = rewriter.create<tensor::ExtractSliceOp>(
          loc, outSliceTy, join, idx({0, 0, 0, g * outPerGroup}),
          idx({batch, outRows, outCols, outPerGroup}), unit);
      Value part = rewriter
                       .create<linalg::Conv2DNhwcHwcfOp>(
                           loc, TypeRange{outSliceTy}, ValueRange{input, hwcf},
                           ValueRange{init}, conv.getStrides(),
                           conv.getDilations())
                       .getResult(0);
      parts.push_back(part);
    }
    join = rewriter.create<tensor::ConcatOp>(loc, /*dim=*/3, parts);

    // Back the way it came in, so the rest of the function is untouched.
    Value nchw = transpose(join, {batch, filters, outRows, outCols},
                           {0, 3, 1, 2});

    // torch-mlir always collapses the groups straight back out of the result,
    // so replace *that* rather than expanding into it. An expand and a collapse
    // that cancel are not free here: they sit between this convolution's
    // transpose back to NCHW and the next one's transpose to NHWC, and stop the
    // two from cancelling -- which is what keeps the group tails in f32 and out
    // of `conv2d_i8`.
    if (conv->hasOneUse()) {
      auto collapse =
          dyn_cast<tensor::CollapseShapeOp>(*conv->getUsers().begin());
      if (collapse &&
          collapse.getReassociationIndices() ==
              SmallVector<ReassociationIndices>(mergeGroups) &&
          collapse.getType() == cast<RankedTensorType>(nchw.getType())) {
        rewriter.replaceOp(collapse, nchw);
        rewriter.eraseOp(conv);
        return success();
      }
    }
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(conv, outTy, nchw,
                                                       mergeGroups);
    return success();
  }
};

class SplitGroupedConv : public impl::SplitGroupedConvBase<SplitGroupedConv> {
public:
  using impl::SplitGroupedConvBase<SplitGroupedConv>::SplitGroupedConvBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SplitGrouped>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
