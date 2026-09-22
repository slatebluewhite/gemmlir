//===- ConvNchwToNhwcPass.cpp ------------------------------*- C++ -*-===//
//
// Rewrites NCHW convolution and pooling into their NHWC forms.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"
#include "Gemmlir/GemmlirPatterns.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_CONVNCHWTONHWC
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// `dim(result, i) = dim(input, permutation[i])`, which is what linalg.transpose
/// means. NCHW -> NHWC keeps N, then takes H, W and finally C.
static constexpr int64_t kNchwToNhwc[] = {0, 2, 3, 1};
static constexpr int64_t kNhwcToNchw[] = {0, 3, 1, 2};
/// FCHW -> HWCF, the layout `linalg.conv_2d_nhwc_hwcf` reads a filter in.
static constexpr int64_t kFchwToHwcf[] = {2, 3, 1, 0};

static Value transposeTo(PatternRewriter &rewriter, Location loc, Value v,
                         ArrayRef<int64_t> perm) {
  auto ty = llvm::cast<RankedTensorType>(v.getType());
  SmallVector<int64_t> shape;
  for (int64_t p : perm)
    shape.push_back(ty.getShape()[p]);
  Value init = rewriter.create<tensor::EmptyOp>(loc, shape, ty.getElementType());
  return rewriter.create<linalg::TransposeOp>(loc, v, init, perm)->getResult(0);
}

static bool allStatic(ValueRange vs) {
  return llvm::all_of(vs, [](Value v) {
    auto t = llvm::dyn_cast<RankedTensorType>(v.getType());
    return t && t.hasStaticShape();
  });
}

class ConvToNhwc : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp conv,
                                PatternRewriter &rewriter) const final {
    if (conv.getInputs().size() != 2 || conv.getOutputs().size() != 1)
      return failure();
    if (!allStatic(conv.getInputs()) || !allStatic(conv.getOutputs()))
      return failure();

    Location loc = conv.getLoc();
    Value input = transposeTo(rewriter, loc, conv.getInputs()[0], kNchwToNhwc);
    Value filter = transposeTo(rewriter, loc, conv.getInputs()[1], kFchwToHwcf);
    Value init = transposeTo(rewriter, loc, conv.getOutputs()[0], kNchwToNhwc);

    auto nhwc = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
        loc, TypeRange{init.getType()}, ValueRange{input, filter},
        ValueRange{init}, conv.getStrides(), conv.getDilations());
    rewriter.replaceOp(conv, transposeTo(rewriter, loc, nhwc->getResult(0),
                                         kNhwcToNchw));
    return success();
  }
};

/// FCHW -> HWCF for a dense filter; a depthwise one has no F, so (C, KH, KW)
/// becomes (KH, KW, C).
static constexpr int64_t kChwToHwc[] = {1, 2, 0};

class DepthwiseConvToNhwc
    : public OpRewritePattern<linalg::DepthwiseConv2DNchwChwOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNchwChwOp conv,
                                PatternRewriter &rewriter) const final {
    if (conv.getInputs().size() != 2 || conv.getOutputs().size() != 1)
      return failure();
    if (!allStatic(conv.getInputs()) || !allStatic(conv.getOutputs()))
      return failure();

    Location loc = conv.getLoc();
    Value input = transposeTo(rewriter, loc, conv.getInputs()[0], kNchwToNhwc);
    Value filter = transposeTo(rewriter, loc, conv.getInputs()[1], kChwToHwc);
    Value init = transposeTo(rewriter, loc, conv.getOutputs()[0], kNchwToNhwc);

    auto nhwc = rewriter.create<linalg::DepthwiseConv2DNhwcHwcOp>(
        loc, TypeRange{init.getType()}, ValueRange{input, filter},
        ValueRange{init}, conv.getStrides(), conv.getDilations());
    rewriter.replaceOp(conv, transposeTo(rewriter, loc, nhwc->getResult(0),
                                         kNhwcToNchw));
    return success();
  }
};

template <typename NchwOp, typename NhwcOp>
class PoolToNhwc : public OpRewritePattern<NchwOp> {
public:
  using OpRewritePattern<NchwOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NchwOp pool,
                                PatternRewriter &rewriter) const final {
    if (pool.getInputs().size() != 2 || pool.getOutputs().size() != 1)
      return failure();
    if (!allStatic(pool.getInputs()) || !allStatic(pool.getOutputs()))
      return failure();

    Location loc = pool.getLoc();
    Value input = transposeTo(rewriter, loc, pool.getInputs()[0], kNchwToNhwc);
    Value init = transposeTo(rewriter, loc, pool.getOutputs()[0], kNchwToNhwc);

    auto nhwc = rewriter.create<NhwcOp>(
        loc, TypeRange{init.getType()},
        ValueRange{input, pool.getInputs()[1]}, ValueRange{init},
        pool.getStrides(), pool.getDilations());
    rewriter.replaceOp(pool, transposeTo(rewriter, loc, nhwc->getResult(0),
                                         kNhwcToNchw));
    return success();
  }
};

/// A `linalg.generic` that only reads a constant, at indices it computes from
/// its own loop indices, is a constant.
///
/// MLIR's `populateConstantFoldLinalgOperations` does not reach this shape: it
/// folds an elementwise operation over constant *operands*, and this one has no
/// operands at all -- it gathers, with `linalg.index` and a `tensor.extract`.
/// That is how torch-mlir writes the kernel a transposed convolution needs,
/// which is the forward kernel with its two spatial axes reflected and its
/// channels swapped. Left alone it is recomputed every inference, and -- worse
/// -- the filter is no longer a compile-time constant, so
/// `--force-quantized-matmul` has no range for it and quantizes it at the
/// *activation's* scale.
///
/// Each value in the body is carried as a constant or as an affine form in the
/// loop indices; anything else, and any operation outside the small vocabulary
/// evaluated here, is refused.
class FoldConstantGather : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.getInputs().empty() || generic.getOutputs().size() != 1 ||
        generic->getNumResults() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    auto resTy = llvm::dyn_cast<RankedTensorType>(generic->getResult(0).getType());
    if (!resTy || !resTy.hasStaticShape() || resTy.getNumElements() > (1 << 20))
      return failure();

    // Every element written exactly once, so the result really is the gather.
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 1 || !maps[0].isPermutation())
      return failure();
    unsigned rank = maps[0].getNumDims();

    Block &body = generic.getRegion().front();
    if (!body.getArguments().back().use_empty())
      return failure();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    auto extract = yield.getOperand(0).getDefiningOp<tensor::ExtractOp>();
    if (!extract)
      return failure();
    DenseElementsAttr source;
    if (!matchPattern(extract.getTensor(), m_Constant(&source)))
      return failure();
    auto srcTy = llvm::dyn_cast<RankedTensorType>(extract.getTensor().getType());
    if (!srcTy || !srcTy.hasStaticShape() ||
        srcTy.getElementType() != resTy.getElementType())
      return failure();
    if (extract.getIndices().size() != static_cast<size_t>(srcTy.getRank()))
      return failure();

    // The loop bounds: the map is a permutation, so each iteration dimension
    // takes its extent from the result dimension it writes.
    SmallVector<int64_t> bounds(rank, 0);
    for (unsigned k = 0; k < rank; k++) {
      auto dim = llvm::dyn_cast<AffineDimExpr>(maps[0].getResult(k));
      if (!dim)
        return failure();
      bounds[dim.getPosition()] = resTy.getDimSize(k);
    }

    // Each value in the body as `c0 + sum c_i * index_i`.
    struct Affine {
      int64_t constant = 0;
      SmallVector<int64_t> coefficients;
    };
    DenseMap<Value, Affine> values;
    auto get = [&](Value v, Affine *out) {
      auto it = values.find(v);
      if (it != values.end()) {
        *out = it->second;
        return true;
      }
      APInt c;
      if (matchPattern(v, m_ConstantInt(&c))) {
        *out = Affine{c.getSExtValue(), SmallVector<int64_t>(rank, 0)};
        return true;
      }
      return false;
    };
    for (Operation &op : body.without_terminator()) {
      if (llvm::isa<arith::ConstantOp>(&op))
        continue;
      if (&op == extract.getOperation())
        continue;
      Affine r;
      r.coefficients.assign(rank, 0);
      if (auto idx = llvm::dyn_cast<linalg::IndexOp>(&op)) {
        if (idx.getDim() >= rank)
          return failure();
        r.coefficients[idx.getDim()] = 1;
        values[idx.getResult()] = r;
        continue;
      }
      Affine x, y;
      if (op.getNumOperands() != 2 || op.getNumResults() != 1 ||
          !get(op.getOperand(0), &x) || !get(op.getOperand(1), &y))
        return failure();
      if (llvm::isa<arith::AddIOp>(&op)) {
        r.constant = x.constant + y.constant;
        for (unsigned i = 0; i < rank; i++)
          r.coefficients[i] = x.coefficients[i] + y.coefficients[i];
      } else if (llvm::isa<arith::SubIOp>(&op)) {
        r.constant = x.constant - y.constant;
        for (unsigned i = 0; i < rank; i++)
          r.coefficients[i] = x.coefficients[i] - y.coefficients[i];
      } else if (llvm::isa<arith::MulIOp>(&op)) {
        bool xConst = llvm::all_of(x.coefficients, [](int64_t c) { return c == 0; });
        bool yConst = llvm::all_of(y.coefficients, [](int64_t c) { return c == 0; });
        if (!xConst && !yConst)
          return failure();
        const Affine &var = xConst ? y : x;
        int64_t k = xConst ? x.constant : y.constant;
        r.constant = var.constant * k;
        for (unsigned i = 0; i < rank; i++)
          r.coefficients[i] = var.coefficients[i] * k;
      } else {
        return failure();
      }
      values[op.getResult(0)] = r;
    }

    SmallVector<Affine> indexForms;
    for (Value v : extract.getIndices()) {
      Affine a;
      if (!get(v, &a))
        return failure();
      indexForms.push_back(a);
    }

    // Walk the iteration space and read the constant.
    SmallVector<Attribute> flat(resTy.getNumElements());
    auto contents = source.getValues<Attribute>();
    SmallVector<int64_t> point(rank, 0);
    int64_t total = 1;
    for (int64_t b : bounds) {
      if (b <= 0)
        return failure();
      total *= b;
    }
    for (int64_t n = 0; n < total; n++) {
      int64_t rest = n;
      for (unsigned d = rank; d-- > 0;) {
        point[d] = rest % bounds[d];
        rest /= bounds[d];
      }
      int64_t from = 0;
      for (auto [axis, form] : llvm::enumerate(indexForms)) {
        int64_t v = form.constant;
        for (unsigned i = 0; i < rank; i++)
          v += form.coefficients[i] * point[i];
        if (v < 0 || v >= srcTy.getDimSize(axis))
          return failure();
        from = from * srcTy.getDimSize(axis) + v;
      }
      int64_t to = 0;
      for (unsigned k = 0; k < rank; k++) {
        auto dim = llvm::cast<AffineDimExpr>(maps[0].getResult(k));
        to = to * resTy.getDimSize(k) + point[dim.getPosition()];
      }
      flat[to] = contents[from];
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        generic, resTy, DenseElementsAttr::get(resTy, flat));
    return success();
  }
};

/// A destination whose value is never read is only supplying a shape.
///
/// Absorbing a transpose into an elementwise operation rewrites the operand's
/// map and leaves the transpose that used to feed its *destination* behind --
/// still a use of the convolution's result, so the result no longer looks
/// single-use and `--fold-batch-norm` will not touch it, and still a real
/// transpose of the whole activation at run time. Pointing the destination at a
/// fresh `tensor.empty` makes it dead.
///
/// Sound when the operation writes every element it is handed: all-parallel
/// iterators and an identity map onto the result.
class DeadDestination : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getOutputs().size() != 1 || generic->getNumResults() != 1)
      return failure();
    Value dest = generic.getOutputs()[0];
    if (dest.getDefiningOp<tensor::EmptyOp>())
      return failure();
    auto destTy = llvm::dyn_cast<RankedTensorType>(dest.getType());
    if (!destTy || !destTy.hasStaticShape())
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    if (!generic.getIndexingMapsArray().back().isIdentity())
      return failure();
    if (!generic.getRegion().front().getArguments().back().use_empty())
      return failure();

    Value empty = rewriter.create<tensor::EmptyOp>(
        generic.getLoc(), destTy.getShape(), destTy.getElementType());
    rewriter.modifyOpInPlace(generic, [&] {
      generic.getOutputsMutable()[0].assign(empty);
    });
    return success();
  }
};

/// True when this transpose is the flatten in front of a linear layer whose
/// weights are constant -- the shape MoveTransposeIntoWeights removes outright.
static bool feedsAClassifier(linalg::TransposeOp transpose) {
  if (!transpose->hasOneUse())
    return false;
  auto collapse =
      llvm::dyn_cast<tensor::CollapseShapeOp>(*transpose->getUsers().begin());
  if (!collapse || !collapse->hasOneUse())
    return false;
  auto matmul = llvm::dyn_cast<linalg::MatmulOp>(*collapse->getUsers().begin());
  return matmul && matmul.getInputs().size() == 2 &&
         matmul.getInputs()[0] == collapse.getResult() &&
         matmul.getInputs()[1].getDefiningOp<arith::ConstantOp>();
}

/// `transpose(elementwise(x))` is `elementwise(x)` over the permuted iteration
/// space -- which is what lets the transpose a convolution leaves behind meet
/// the one the next convolution puts in front, so `--canonicalize` can cancel
/// the pair. Nothing else moves them: the relu between two layers sits exactly
/// in the way.
///
/// The operands are untouched; only the maps are composed with the permutation,
/// so a bias broadcast stays a broadcast of the same small constant.
class PushTransposeThroughElementwise
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    // MoveTransposeIntoWeights gets rid of this one entirely rather than moving
    // it, so leave the flatten in front of a classifier to it.
    if (feedsAClassifier(transpose))
      return failure();
    auto generic = transpose.getInput().getDefiningOp<linalg::GenericOp>();
    if (!generic || !generic->hasOneUse() || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    // The destination has to be ours to replace, and the body must not read it.
    if (!generic.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
      return failure();
    Block &body = generic.getRegion().front();
    if (!body.getArguments().back().use_empty())
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != generic.getInputs().size() + 1 || !maps.back().isIdentity())
      return failure();

    auto resTy = llvm::dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
    if (!resTy || !resTy.hasStaticShape())
      return failure();

    // Result iteration d_k reads the generic at the index whose position
    // permutation[k] is d_k, i.e. through the inverse permutation.
    ArrayRef<int64_t> perm = transpose.getPermutation();
    unsigned rank = perm.size();
    SmallVector<AffineExpr> through(rank);
    for (unsigned k = 0; k < rank; k++)
      through[perm[k]] = rewriter.getAffineDimExpr(k);
    AffineMap toOld = AffineMap::get(rank, 0, through, rewriter.getContext());

    SmallVector<AffineMap> newMaps;
    for (unsigned i = 0; i < generic.getInputs().size(); i++)
      newMaps.push_back(maps[i].compose(toOld));
    newMaps.push_back(AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    Location loc = generic.getLoc();
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                  resTy.getElementType());
    auto moved = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, generic.getInputs(), ValueRange{init}, newMaps,
        iters);
    rewriter.cloneRegionBefore(generic.getRegion(), moved.getRegion(),
                               moved.getRegion().begin());
    rewriter.replaceOp(transpose, moved.getResults());
    return success();
  }
};

/// An elementwise operand that is a `transpose` is the operand itself, read
/// through the permutation.
///
/// This is the input-side half. Together with pushing a transpose past the
/// result, it walks both transposes a layer leaves behind into the elementwise
/// operation between the layers, where they compose to the identity and
/// disappear: the relu ends up reading the previous convolution's NHWC result
/// directly.
class AbsorbTransposeIntoElementwise : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != generic.getInputs().size() + 1)
      return failure();

    SmallVector<Value> inputs(generic.getInputs());
    bool changed = false;
    for (unsigned i = 0; i < inputs.size(); i++) {
      auto transpose = inputs[i].getDefiningOp<linalg::TransposeOp>();
      if (!transpose)
        continue;
      ArrayRef<int64_t> perm = transpose.getPermutation();
      if (maps[i].getNumResults() != perm.size())
        continue;
      // The transpose reads its input at r, where r[permutation[k]] = q_k.
      unsigned rank = perm.size();
      SmallVector<AffineExpr> through(rank);
      for (unsigned k = 0; k < rank; k++)
        through[perm[k]] = rewriter.getAffineDimExpr(k);
      AffineMap toSource = AffineMap::get(rank, 0, through, rewriter.getContext());
      maps[i] = toSource.compose(maps[i]);
      inputs[i] = transpose.getInput();
      changed = true;
    }
    if (!changed)
      return failure();

    auto moved = rewriter.create<linalg::GenericOp>(
        generic.getLoc(), generic.getResultTypes(), inputs,
        generic.getOutputs(), maps, generic.getIteratorTypesArray());
    rewriter.cloneRegionBefore(generic.getRegion(), moved.getRegion(),
                               moved.getRegion().begin());
    rewriter.replaceOp(generic, moved.getResults());
    return success();
  }
};

/// A transposed constant is a constant. The filters are the whole reason this
/// matters: moving to NHWC permutes them, and leaving that as an operation
/// would run it on every inference -- and would also hide the constant from the
/// reshape folding that img2col relies on, so the weights would stop being
/// compile-time i8 globals.
class FoldTransposeOfConstant : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    auto cst = transpose.getInput().getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return failure();
    auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
    auto srcTy = llvm::dyn_cast<RankedTensorType>(transpose.getInput().getType());
    auto resTy = llvm::dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
    if (!dense || !srcTy || !resTy || !srcTy.hasStaticShape() ||
        !resTy.hasStaticShape())
      return failure();
    if (dense.isSplat()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          transpose, resTy,
          DenseElementsAttr::get(resTy, dense.getSplatValue<Attribute>()));
      return success();
    }

    ArrayRef<int64_t> perm = transpose.getPermutation();
    ArrayRef<int64_t> srcShape = srcTy.getShape();
    ArrayRef<int64_t> resShape = resTy.getShape();
    unsigned rank = perm.size();

    SmallVector<Attribute> values(dense.getValues<Attribute>());
    SmallVector<Attribute> out;
    out.reserve(resTy.getNumElements());
    SmallVector<int64_t> idx(rank, 0);
    for (int64_t linear = 0, n = resTy.getNumElements(); linear < n; linear++) {
      int64_t rem = linear;
      for (int64_t d = rank - 1; d >= 0; d--) {
        idx[d] = rem % resShape[d];
        rem /= resShape[d];
      }
      // dim(result, k) = dim(input, permutation[k]), so the source index at
      // position permutation[k] is the destination's k-th.
      SmallVector<int64_t> src(rank, 0);
      for (unsigned k = 0; k < rank; k++)
        src[perm[k]] = idx[k];
      int64_t srcLinear = 0;
      for (unsigned d = 0; d < rank; d++)
        srcLinear = srcLinear * srcShape[d] + src[d];
      out.push_back(values[srcLinear]);
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        transpose, resTy, DenseElementsAttr::get(resTy, out));
    return success();
  }
};

/// The transpose in front of a classifier belongs in its weights.
///
/// A convolution block ends NHWC and the flatten before the first linear layer
/// expects the frontend's NCHW order, so the layout rewrite leaves one transpose
/// there with nothing to cancel against. It does not have to run: flattening the
/// other order just permutes which weight row each activation meets, and the
/// weights are constants. Reshaping them to the spatial extents, permuting, and
/// flattening back is folded here, so the transpose disappears entirely.
class MoveTransposeIntoWeights : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const final {
    if (matmul.getInputs().size() != 2 || matmul->getNumResults() != 1)
      return failure();
    auto flat = matmul.getInputs()[0].getDefiningOp<tensor::CollapseShapeOp>();
    if (!flat || !flat->hasOneUse())
      return failure();
    auto transpose = flat.getSrc().getDefiningOp<linalg::TransposeOp>();
    if (!transpose || !transpose->hasOneUse())
      return failure();

    // Only worth it when the weights are constant: permuting them is free at
    // compile time, and at run time it would move K x N elements to save K.
    if (!matmul.getInputs()[1].getDefiningOp<arith::ConstantOp>())
      return failure();

    auto midTy = llvm::dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
    auto srcTy = llvm::dyn_cast<RankedTensorType>(transpose.getInput().getType());
    auto weightTy = llvm::dyn_cast<RankedTensorType>(matmul.getInputs()[1].getType());
    if (!midTy || !srcTy || !weightTy || !midTy.hasStaticShape() ||
        !srcTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        weightTy.getRank() != 2)
      return failure();

    // Everything but a leading batch is flattened into the contracted axis.
    ArrayRef<int64_t> perm = transpose.getPermutation();
    unsigned rank = perm.size();
    if (rank < 2 || perm[0] != 0)
      return failure();
    SmallVector<ReassociationIndices> groups = flat.getReassociationIndices();
    if (groups.size() != 2 || groups[0].size() != 1 || groups[0][0] != 0 ||
        groups[1].size() != rank - 1)
      return failure();
    int64_t contracted = 1;
    for (unsigned d = 1; d < rank; d++)
      contracted *= midTy.getShape()[d];
    if (contracted != weightTy.getShape()[0])
      return failure();

    // Weight row `i` currently belongs to the transposed dimensions in order.
    // Put it where the untransposed ones are: result dimension k walks input
    // dimension perm[k], so input dimension d sits at the k with perm[k] == d.
    SmallVector<int64_t> toInput(rank, 0);
    for (unsigned d = 1; d < rank; d++) {
      unsigned k = 0;
      for (; k < rank; k++)
        if ((unsigned)perm[k] == d)
          break;
      if (k == rank)
        return failure();
      toInput[d - 1] = k - 1;
    }
    toInput[rank - 1] = rank - 1; // the output channels stay last

    Location loc = matmul.getLoc();
    SmallVector<int64_t> wide;
    for (unsigned d = 1; d < rank; d++)
      wide.push_back(midTy.getShape()[d]);
    wide.push_back(weightTy.getShape()[1]);
    SmallVector<ReassociationIndices> weightGroups;
    ReassociationIndices merged;
    for (unsigned d = 0; d + 1 < wide.size(); d++)
      merged.push_back(d);
    weightGroups.push_back(merged);
    weightGroups.push_back({(int64_t)wide.size() - 1});

    Value expanded = rewriter.create<tensor::ExpandShapeOp>(
        loc, RankedTensorType::get(wide, weightTy.getElementType()),
        matmul.getInputs()[1], weightGroups);
    Value reordered = transposeTo(rewriter, loc, expanded, toInput);
    Value weights = rewriter.create<tensor::CollapseShapeOp>(loc, weightTy,
                                                            reordered,
                                                            weightGroups);
    Value flatSrc = rewriter.create<tensor::CollapseShapeOp>(
        loc, flat.getType(), transpose.getInput(), groups);

    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        matmul, matmul->getResultTypes(), ValueRange{flatSrc, weights},
        matmul.getOutputs());
    return success();
  }
};

/// Reads `m` as a permutation of `rank` dimensions, accepting the constant 0 a
/// frontend writes where an axis has extent 1.
static bool asPermutation(AffineMap m, unsigned rank,
                          SmallVectorImpl<int64_t> &perm) {
  if (m.getNumResults() != rank || m.getNumDims() != rank)
    return false;
  perm.assign(rank, -1);
  llvm::SmallDenseSet<int64_t> used;
  SmallVector<unsigned> constants;
  for (auto [r, e] : llvm::enumerate(m.getResults())) {
    if (auto dim = llvm::dyn_cast<AffineDimExpr>(e)) {
      perm[r] = dim.getPosition();
      if (!used.insert(perm[r]).second)
        return false;
      continue;
    }
    auto cst = llvm::dyn_cast<AffineConstantExpr>(e);
    if (!cst || cst.getValue() != 0)
      return false;
    constants.push_back(r);
  }
  // Whatever dimension is left over is the one the constant stands in for.
  for (unsigned r : constants) {
    int64_t free = -1;
    for (unsigned d = 0; d < rank; d++)
      if (!used.contains((int64_t)d)) {
        free = d;
        break;
      }
    if (free < 0)
      return false;
    perm[r] = free;
    used.insert(free);
  }
  return used.size() == rank;
}

/// The same, for a permutation that has ended up inside an elementwise
/// operation's maps rather than standing on its own.
///
/// With a relu between the last convolution and the flatten there is no
/// `linalg.transpose` left to match: absorbing it into the relu gets there
/// first, and the relu is then the thing producing the frontend's layout. It
/// can be rewritten to produce the convolution's layout instead -- every map
/// composed with the inverse permutation, which makes the convolution's own
/// read the identity -- and the classifier's weight rows permuted to match.
class MovePermutedElementwiseIntoWeights
    : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const final {
    if (matmul.getInputs().size() != 2 || matmul->getNumResults() != 1)
      return failure();
    if (!matmul.getInputs()[1].getDefiningOp<arith::ConstantOp>())
      return failure();
    auto flat = matmul.getInputs()[0].getDefiningOp<tensor::CollapseShapeOp>();
    if (!flat || !flat->hasOneUse())
      return failure();
    auto generic = flat.getSrc().getDefiningOp<linalg::GenericOp>();
    if (!generic || !generic->hasOneUse() || generic.getOutputs().size() != 1 ||
        generic->getNumResults() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    if (!generic.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
      return failure();
    if (!generic.getRegion().front().getArguments().back().use_empty())
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != generic.getInputs().size() + 1 || !maps.back().isIdentity())
      return failure();
    unsigned rank = maps.back().getNumDims();

    // Exactly one operand reads the whole iteration space through a
    // permutation: that is the layer's result, and its layout is the one to
    // adopt. Anything else is a broadcast and just comes along.
    int anchor = -1;
    SmallVector<int64_t> p;
    for (auto [i, m] : llvm::enumerate(llvm::ArrayRef(maps).drop_back())) {
      SmallVector<int64_t> candidate;
      if (!asPermutation(m, rank, candidate))
        continue;
      if (llvm::equal(candidate, llvm::seq<int64_t>(0, rank)))
        return failure(); // already in the producer's layout; nothing to move
      if (anchor >= 0)
        return failure();
      anchor = i;
      p = candidate;
    }
    if (anchor < 0 || p[0] != 0)
      return failure();

    auto oldTy = llvm::dyn_cast<RankedTensorType>(generic->getResult(0).getType());
    auto newTy = llvm::dyn_cast<RankedTensorType>(generic.getInputs()[anchor].getType());
    auto weightTy = llvm::dyn_cast<RankedTensorType>(matmul.getInputs()[1].getType());
    if (!oldTy || !newTy || !weightTy || !oldTy.hasStaticShape() ||
        !newTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        weightTy.getRank() != 2 || (unsigned)oldTy.getRank() != rank)
      return failure();

    SmallVector<ReassociationIndices> groups = flat.getReassociationIndices();
    if (groups.size() != 2 || groups[0].size() != 1 || groups[0][0] != 0 ||
        groups[1].size() != rank - 1)
      return failure();
    int64_t contracted = 1;
    for (unsigned d = 1; d < rank; d++)
      contracted *= oldTy.getShape()[d];
    if (contracted != weightTy.getShape()[0])
      return failure();

    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> permExprs, inverseExprs(rank);
    for (unsigned k = 0; k < rank; k++) {
      permExprs.push_back(rewriter.getAffineDimExpr(p[k]));
      inverseExprs[p[k]] = rewriter.getAffineDimExpr(k);
    }
    AffineMap inverse = AffineMap::get(rank, 0, inverseExprs, ctx);

    Location loc = matmul.getLoc();
    SmallVector<AffineMap> newMaps;
    for (AffineMap m : llvm::ArrayRef(maps).drop_back())
      newMaps.push_back(m.compose(inverse));
    newMaps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

    auto resTy = RankedTensorType::get(newTy.getShape(), oldTy.getElementType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                  resTy.getElementType());
    auto moved = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, generic.getInputs(), ValueRange{init}, newMaps,
        iters);
    rewriter.cloneRegionBefore(generic.getRegion(), moved.getRegion(),
                               moved.getRegion().begin());

    // Old flattened position (i_1..i_{rank-1}) becomes (i_{p_1}..i_{p_{rank-1}}),
    // so the weight's rows are reshaped to the old extents and permuted the same
    // way before being flattened back.
    SmallVector<int64_t> wide;
    for (unsigned d = 1; d < rank; d++)
      wide.push_back(oldTy.getShape()[d]);
    wide.push_back(weightTy.getShape()[1]);
    SmallVector<ReassociationIndices> weightGroups;
    ReassociationIndices merged;
    for (unsigned d = 0; d + 1 < wide.size(); d++)
      merged.push_back(d);
    weightGroups.push_back(merged);
    weightGroups.push_back({(int64_t)wide.size() - 1});
    SmallVector<int64_t> wperm;
    for (unsigned k = 1; k < rank; k++)
      wperm.push_back(p[k] - 1);
    wperm.push_back(rank - 1);

    Value expanded = rewriter.create<tensor::ExpandShapeOp>(
        loc, RankedTensorType::get(wide, weightTy.getElementType()),
        matmul.getInputs()[1], weightGroups);
    Value reordered = transposeTo(rewriter, loc, expanded, wperm);
    Value weights = rewriter.create<tensor::CollapseShapeOp>(loc, weightTy,
                                                            reordered,
                                                            weightGroups);
    Value flatSrc = rewriter.create<tensor::CollapseShapeOp>(
        loc, flat.getType(), moved->getResult(0), groups);

    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        matmul, matmul->getResultTypes(), ValueRange{flatSrc, weights},
        matmul.getOutputs());
    return success();
  }
};

/// `transpose(fill(c))` is `fill(c)` in the transposed shape.
class PushTransposeThroughFill : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    auto fill = transpose.getInput().getDefiningOp<linalg::FillOp>();
    if (!fill || !fill->hasOneUse() || fill.getInputs().size() != 1)
      return failure();
    auto resTy = llvm::dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
    if (!resTy || !resTy.hasStaticShape())
      return failure();
    Value init = rewriter.create<tensor::EmptyOp>(
        transpose.getLoc(), resTy.getShape(), resTy.getElementType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(transpose, fill.getInputs(),
                                                ValueRange{init});
    return success();
  }
};

class ConvNchwToNhwc : public impl::ConvNchwToNhwcBase<ConvNchwToNhwc> {
public:
  using impl::ConvNchwToNhwcBase<ConvNchwToNhwc>::ConvNchwToNhwcBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvToNhwc, DeadDestination, DepthwiseConvToNhwc,
                 FoldConstantGather,
                 PoolToNhwc<linalg::PoolingNchwMaxOp, linalg::PoolingNhwcMaxOp>,
                 PoolToNhwc<linalg::PoolingNchwSumOp, linalg::PoolingNhwcSumOp>,
                 PushTransposeThroughElementwise,
                 PushTransposeThroughFill, FoldTransposeOfConstant,
                 MoveTransposeIntoWeights,
                 MovePermutedElementwiseIntoWeights>(&getContext());
    populateAbsorbTransposePatterns(patterns);
    linalg::TransposeOp::getCanonicalizationPatterns(patterns, &getContext());
    // A frontend hands weights over behind a transposing `linalg.generic`, and
    // moving to NHWC puts more reshaping in front of them. Folding the constants
    // here keeps all of it at compile time -- and keeps the weights foldable to
    // i8 globals later, which only works from a plain constant.
    linalg::populateConstantFoldLinalgOperations(
        patterns, [](OpOperand *) { return true; });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populateAbsorbTransposePatterns(RewritePatternSet &patterns) {
  patterns.add<AbsorbTransposeIntoElementwise>(patterns.getContext());
}

} // namespace mlir::gemmlir
