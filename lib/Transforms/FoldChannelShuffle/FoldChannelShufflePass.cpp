//===- FoldChannelShufflePass.cpp -----------------------------*- C++ -*-===//
//
// A channel permutation belongs in the weights that read it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FOLDCHANNELSHUFFLE
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The one axis a reshape splits or merges, if it is exactly one.
///
/// A reassociation is a list of groups of source dimensions. Every group of one
/// leaves its dimension alone; the axis being worked on is the single group of
/// two, and there has to be exactly one of those.
std::optional<unsigned> soleSplitAxis(ArrayRef<ReassociationIndices> groups) {
  std::optional<unsigned> axis;
  for (auto [i, group] : llvm::enumerate(groups)) {
    if (group.size() == 1)
      continue;
    if (group.size() != 2 || axis)
      return std::nullopt;
    axis = i;
  }
  return axis;
}

/// A transpose of exactly two adjacent axes, written either as
/// `linalg.transpose` or as the copying `linalg.generic` a frontend emits.
bool swapsAdjacent(Operation *op, unsigned first, Value *input) {
  auto isSwap = [&](ArrayRef<int64_t> perm) {
    for (unsigned i = 0; i < perm.size(); i++) {
      int64_t want = i == first ? first + 1 : (i == first + 1 ? first : i);
      if (perm[i] != want)
        return false;
    }
    return true;
  };
  if (auto transpose = dyn_cast_or_null<linalg::TransposeOp>(op)) {
    if (!isSwap(transpose.getPermutation()))
      return false;
    *input = transpose.getInput();
    return true;
  }
  auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1)
    return false;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 2)
    return false;
  Block &body = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != body.getArgument(0))
    return false;
  // A frontend writes the swap on either side: iterate over the input and
  // write permuted, or iterate over the result and read permuted. Exchanging
  // two axes is its own inverse, so either says the same thing.
  auto permutationOf = [&](AffineMap m, SmallVector<int64_t> *perm) {
    for (AffineExpr e : m.getResults()) {
      auto dim = dyn_cast<AffineDimExpr>(e);
      if (!dim)
        return false;
      perm->push_back(dim.getPosition());
    }
    return true;
  };
  SmallVector<int64_t> read, write;
  if (!permutationOf(maps[0], &read) || !permutationOf(maps[1], &write))
    return false;
  bool readsInOrder = maps[0].isIdentity(), writesInOrder = maps[1].isIdentity();
  if (readsInOrder == writesInOrder)
    return false;
  if (!isSwap(readsInOrder ? write : read))
    return false;
  *input = generic.getInputs()[0];
  return true;
}

/// `collapse(swap(expand(x)))` on one axis is a permutation of that axis.
/// Splitting it into `g` groups of `m` and swapping puts input channel
/// `i*m + j` at output channel `j*g + i`.
std::optional<SmallVector<int64_t>> channelShuffleOf(Value v, unsigned axis,
                                                     Value *source,
                                                     OpOperand **rewire) {
  // A convolution reads its input through the border it needs, and padding the
  // two spatial axes with a constant commutes with permuting the channels --
  // so the shuffle may be under one, and it is that operand that gets rewired.
  while (auto pad = v.getDefiningOp<tensor::PadOp>()) {
    if (!pad->hasOneUse())
      break;
    SmallVector<OpFoldResult> low = pad.getMixedLowPad();
    SmallVector<OpFoldResult> high = pad.getMixedHighPad();
    if (axis >= low.size())
      return std::nullopt;
    auto isZero = [](OpFoldResult v) {
      auto attr = dyn_cast<Attribute>(v);
      return attr && cast<IntegerAttr>(attr).getInt() == 0;
    };
    if (!isZero(low[axis]) || !isZero(high[axis]))
      return std::nullopt;
    *rewire = &pad.getSourceMutable();
    v = pad.getSource();
  }
  auto collapse = v.getDefiningOp<tensor::CollapseShapeOp>();
  if (!collapse || !collapse->hasOneUse())
    return std::nullopt;
  std::optional<unsigned> merged = soleSplitAxis(collapse.getReassociationIndices());
  if (!merged || *merged != axis)
    return std::nullopt;

  Value swapped;
  Operation *producer = collapse.getSrc().getDefiningOp();
  if (!producer || !producer->hasOneUse() || !swapsAdjacent(producer, axis, &swapped))
    return std::nullopt;

  auto expand = swapped.getDefiningOp<tensor::ExpandShapeOp>();
  if (!expand || !expand->hasOneUse())
    return std::nullopt;
  std::optional<unsigned> split = soleSplitAxis(expand.getReassociationIndices());
  if (!split || *split != axis)
    return std::nullopt;

  auto expandedTy = cast<RankedTensorType>(expand.getResult().getType());
  int64_t g = expandedTy.getDimSize(axis), m = expandedTy.getDimSize(axis + 1);
  if (g < 1 || m < 1)
    return std::nullopt;
  auto srcTy = cast<RankedTensorType>(expand.getSrc().getType());
  if (srcTy.getDimSize(axis) != g * m)
    return std::nullopt;

  SmallVector<int64_t> perm(g * m);
  for (int64_t i = 0; i < g; i++)
    for (int64_t j = 0; j < m; j++)
      perm[j * g + i] = i * m + j;
  *source = expand.getSrc();
  return perm;
}

/// Permutes one axis of a constant.
DenseElementsAttr permuteAxis(DenseElementsAttr values, RankedTensorType type,
                              unsigned axis, ArrayRef<int64_t> perm) {
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < type.getRank(); d++)
    inner *= type.getDimSize(d);
  int64_t extent = type.getDimSize(axis);
  int64_t outer = type.getNumElements() / (extent * inner);

  SmallVector<Attribute> flat(values.getValues<Attribute>());
  SmallVector<Attribute> out(flat.size());
  for (int64_t o = 0; o < outer; o++)
    for (int64_t c = 0; c < extent; c++)
      for (int64_t k = 0; k < inner; k++)
        out[(o * extent + c) * inner + k] =
            flat[(o * extent + perm[c]) * inner + k];
  return DenseElementsAttr::get(type, out);
}

/// A convolution reading a permuted input computes the same thing as the same
/// convolution reading the original with its filter's input channels permuted
/// the other way -- and the filter is a constant, so the permutation is done
/// once at compile time and nothing moves at run time.
template <typename ConvOp, unsigned InputChannelAxis, unsigned FilterChannelAxis>
class FoldIntoFilter : public OpRewritePattern<ConvOp> {
public:
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp conv,
                                PatternRewriter &rewriter) const final {
    if (conv.getInputs().size() != 2)
      return failure();
    Value source;
    OpOperand *rewire = &conv.getInputsMutable()[0];
    std::optional<SmallVector<int64_t>> perm =
        channelShuffleOf(conv.getInputs()[0], InputChannelAxis, &source, &rewire);
    if (!perm)
      return failure();

    Value filter = conv.getInputs()[1];
    auto filterTy = dyn_cast<RankedTensorType>(filter.getType());
    DenseElementsAttr values;
    if (!filterTy || !filterTy.hasStaticShape() ||
        !matchPattern(filter, m_Constant(&values)))
      return failure();
    if (filterTy.getRank() <= static_cast<int64_t>(FilterChannelAxis) ||
        filterTy.getDimSize(FilterChannelAxis) !=
            static_cast<int64_t>(perm->size()))
      return failure();

    // The shuffle reads input channel `perm[k]` into output channel `k`, so
    // the convolution's weight for `k` is the weight the original channel
    // `perm[k]` should meet: the filter takes the **inverse**. The permutation
    // is not an involution -- for two groups of eight, channel 8 goes to 1 but
    // channel 1 goes to 8 only when the groups are the same size as each other
    // -- so getting this the wrong way round is a wrong answer, not a
    // rearranged one. It cost a relative L2 of 0.12 against 0.005.
    SmallVector<int64_t> inverse(perm->size());
    for (auto [k, c] : llvm::enumerate(*perm))
      inverse[c] = k;
    Value permuted = rewriter.create<arith::ConstantOp>(
        conv.getLoc(), filterTy,
        permuteAxis(values, filterTy, FilterChannelAxis, inverse));
    Operation *owner = rewire->getOwner();
    rewriter.modifyOpInPlace(owner, [&] { rewire->assign(source); });
    rewriter.modifyOpInPlace(
        conv, [&] { conv.getInputsMutable()[1].assign(permuted); });
    return success();
  }
};

class FoldChannelShuffle : public impl::FoldChannelShuffleBase<FoldChannelShuffle> {
public:
  using impl::FoldChannelShuffleBase<FoldChannelShuffle>::FoldChannelShuffleBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldIntoFilter<linalg::Conv2DNchwFchwOp, 1, 1>,
                 FoldIntoFilter<linalg::Conv2DNhwcHwcfOp, 3, 2>>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
