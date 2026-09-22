//===- GatherToMemrefCopyPass.cpp -------------------------------*- C++ -*-===//
//
// A copy whose read is a strided view of its source is a `memref.copy`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_GATHERTOMEMREFCOPY
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// How many bytes the innermost packed run has to reach before a `memref.copy`
/// beats the loop nest `--convert-linalg-to-loops` would emit. One `memcpy`
/// call per run has to earn its overhead; a run of a single element never does.
///
/// Measured on the board. A run of 24 bytes (im2col over a contiguous input) is
/// worth 19 ms of ResNeXt's 51; a run of 8 (im2col over one group's slice of a
/// padded buffer, where the channels of the next group sit in between) is worth
/// 1.2 ms of 95; a run of 1 (a transpose) *costs* 0.5 ms of 2.5.
constexpr int64_t kLeastRunBytes = 8;

/// The element offset an affine expression reads, as a stride per iteration
/// dimension plus a constant. Only `+` and multiplication by a constant, which
/// is the whole vocabulary a strided view has; a `floordiv` or a `mod` is not
/// one of these and fails.
static bool linearize(AffineExpr expr, int64_t scale,
                      SmallVectorImpl<int64_t> &perDim, int64_t &constant) {
  if (auto dim = llvm::dyn_cast<AffineDimExpr>(expr)) {
    perDim[dim.getPosition()] += scale;
    return true;
  }
  if (auto cst = llvm::dyn_cast<AffineConstantExpr>(expr)) {
    constant += scale * cst.getValue();
    return true;
  }
  auto bin = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin)
    return false;
  if (bin.getKind() == AffineExprKind::Add)
    return linearize(bin.getLHS(), scale, perDim, constant) &&
           linearize(bin.getRHS(), scale, perDim, constant);
  if (bin.getKind() != AffineExprKind::Mul)
    return false;
  // One side has to be the constant; affine normalizes it to the right.
  auto rhs = llvm::dyn_cast<AffineConstantExpr>(bin.getRHS());
  if (!rhs)
    return false;
  return linearize(bin.getLHS(), scale * rhs.getValue(), perDim, constant);
}

/// The allocation a memref ultimately views, and the byte range of it the
/// memref can touch.
///
/// The range is the point. `--plan-static-buffers` puts every buffer in the
/// function into **one** global arena, so "these are different allocations" is
/// not something the defining operations can be asked -- both sides bottom out
/// at the same `memref.global`. What can be proved is that the two windows do
/// not overlap, and that is what this is for. Conservative: anything it cannot
/// read exactly fails.
static bool byteRange(Value v, Value &base, int64_t &lo, int64_t &hi) {
  auto ty = llvm::dyn_cast<MemRefType>(v.getType());
  if (!ty || !ty.hasStaticShape())
    return false;
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(ty.getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset) ||
      llvm::any_of(strides, ShapedType::isDynamic))
    return false;
  if (!ty.getElementType().isIntOrFloat())
    return false;
  unsigned bits = ty.getElementType().getIntOrFloatBitWidth();
  if (bits % 8 != 0)
    return false;
  int64_t elemBytes = bits / 8;

  // The furthest element any index reaches, plus one.
  int64_t span = 1;
  for (auto [size, stride] : llvm::zip(ty.getShape(), strides)) {
    if (size < 1)
      return false;
    span += (size - 1) * (stride < 0 ? -stride : stride);
  }
  lo = offset * elemBytes;
  hi = lo + span * elemBytes;

  // A view's window is already absolute in its source's frame, except for
  // `memref.view`, which carries its shift as an operand and starts again at
  // offset zero.
  base = v;
  while (Operation *def = base.getDefiningOp()) {
    if (auto view = llvm::dyn_cast<memref::ViewOp>(def)) {
      llvm::APInt shift;
      if (!matchPattern(view.getByteShift(), m_ConstantInt(&shift)))
        return false;
      lo += shift.getSExtValue();
      hi += shift.getSExtValue();
      base = view.getSource();
      continue;
    }
    if (auto sub = llvm::dyn_cast<memref::SubViewOp>(def)) {
      base = sub.getSource();
      continue;
    }
    if (auto cast = llvm::dyn_cast<memref::CastOp>(def)) {
      base = cast.getSource();
      continue;
    }
    if (auto rein = llvm::dyn_cast<memref::ReinterpretCastOp>(def)) {
      base = rein.getSource();
      continue;
    }
    if (auto expand = llvm::dyn_cast<memref::ExpandShapeOp>(def)) {
      base = expand.getSrc();
      continue;
    }
    if (auto collapse = llvm::dyn_cast<memref::CollapseShapeOp>(def)) {
      base = collapse.getSrc();
      continue;
    }
    if (auto get = llvm::dyn_cast<memref::GetGlobalOp>(def)) {
      base = get.getResult();
      break;
    }
    break;
  }
  return true;
}

/// `out[i] = in[map(i)]` with a `map` that is linear in the iteration indices
/// is a copy between two strided views of the same shape, and `memref.copy`
/// says exactly that.
///
/// im2col is the one that matters. Its read is
/// `(n, oh, ow, kh, kw, c) -> (n, oh*sh + kh*dh, ow*sw + kw*dw, c)`, linear in
/// every index, so the pack is one `memref.copy` from a view whose windows
/// overlap -- which is fine, nothing writes through it. The runtime then copies
/// the longest run that is packed on both sides in one `memcpy`: with unit
/// stride and dilation that run is `KW*C`, so a pack that was 18432 scalar
/// stores becomes 768 `memcpy`s of 24 bytes.
///
/// `--convert-linalg-to-loops` would otherwise emit a load and a store per
/// element, and on this board the packs are most of what is left of the
/// grouped models.
class GatherToCopy : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.hasPureBufferSemantics())
      return failure();
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    Block &body = generic.getRegion().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != body.getArgument(0))
      return failure();

    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2)
      return failure();
    // The same copy can arrive written either way round: the permutation on
    // the read with the write straight, or the read straight with the
    // permutation on the write. They differ only in what the loops are named,
    // so relabel them by the inverse of the write's map and carry on with the
    // one shape the rest of this pattern knows. Without it a channel shuffle
    // -- `(n, g, c, h, w)` out of `(n, c, g, h, w)`, whose innermost image is
    // 1024 contiguous bytes on both sides -- stayed a scalar loop.
    if (!maps[1].isIdentity()) {
      if (!maps[1].isPermutation())
        return failure();
      AffineMap relabel = inversePermutation(maps[1]);
      if (!relabel)
        return failure();
      maps[0] = maps[0].compose(relabel);
      maps[1] = maps[1].compose(relabel);
      if (!maps[1].isIdentity())
        return failure();
    }

    Value src = generic.getInputs()[0], dst = generic.getOutputs()[0];
    auto srcTy = llvm::dyn_cast<MemRefType>(src.getType());
    auto dstTy = llvm::dyn_cast<MemRefType>(dst.getType());
    if (!srcTy || !dstTy || !srcTy.hasStaticShape() || !dstTy.hasStaticShape())
      return failure();
    if (srcTy.getElementType() != dstTy.getElementType())
      return failure();
    if (srcTy.getMemorySpace() != dstTy.getMemorySpace())
      return failure();

    // The source view's own windows overlap each other, which is harmless --
    // nothing writes through it. What is not harmless is the source overlapping
    // the *destination*: `memrefCopy` copies whole runs, so where the elementwise
    // walk would have read a value the copy may already have overwritten it.
    // Every buffer here lives in one static arena, so prove the windows apart.
    Value srcBase, dstBase;
    int64_t srcLo = 0, srcHi = 0, dstLo = 0, dstHi = 0;
    if (!byteRange(src, srcBase, srcLo, srcHi) ||
        !byteRange(dst, dstBase, dstLo, dstHi))
      return failure();
    if (srcBase == dstBase && srcLo < dstHi && dstLo < srcHi)
      return failure();

    SmallVector<int64_t> srcStrides;
    int64_t srcOffset = 0;
    if (failed(srcTy.getStridesAndOffset(srcStrides, srcOffset)) ||
        ShapedType::isDynamic(srcOffset) ||
        llvm::any_of(srcStrides, ShapedType::isDynamic))
      return failure();

    unsigned loops = maps[0].getNumDims();
    if (maps[0].getNumResults() != (unsigned)srcTy.getRank() ||
        loops != (unsigned)dstTy.getRank())
      return failure();

    SmallVector<int64_t> viewStrides(loops, 0);
    int64_t viewOffset = srcOffset;
    for (unsigned r = 0; r < maps[0].getNumResults(); r++)
      if (!linearize(maps[0].getResult(r), srcStrides[r], viewStrides,
                     viewOffset))
        return failure();

    // Only when there is a run to copy. `memrefCopy` moves the longest suffix
    // of dimensions that is packed on *both* sides in one `memcpy` and walks
    // the rest an element at a time -- so this pays exactly when that run is
    // long, and loses when it is not. A transpose has no common run at all:
    // measured on the board, im2col's 24-element run took ResNeXt's bare
    // grouped convolution from 51.1 to 32.0 ms, while the i8 transpose in an
    // attention block went the other way, 2.47 to 3.03 ms.
    SmallVector<int64_t> dstStrides;
    int64_t dstOffset = 0;
    if (failed(dstTy.getStridesAndOffset(dstStrides, dstOffset)) ||
        llvm::any_of(dstStrides, ShapedType::isDynamic))
      return failure();
    int64_t run = 1;
    for (int d = (int)loops - 1; d >= 0; d--) {
      if (viewStrides[d] != run || dstStrides[d] != run)
        break;
      run *= dstTy.getDimSize(d);
    }
    unsigned elemBits = srcTy.getElementType().getIntOrFloatBitWidth();
    if (run * (int64_t)(elemBits / 8) < kLeastRunBytes)
      return failure();

    Location loc = generic.getLoc();
    auto viewTy = MemRefType::get(
        dstTy.getShape(), srcTy.getElementType(),
        StridedLayoutAttr::get(rewriter.getContext(), viewOffset, viewStrides),
        srcTy.getMemorySpace());

    SmallVector<OpFoldResult> sizes, strides;
    for (int64_t d : dstTy.getShape())
      sizes.push_back(rewriter.getIndexAttr(d));
    for (int64_t s : viewStrides)
      strides.push_back(rewriter.getIndexAttr(s));
    Value view = rewriter.create<memref::ReinterpretCastOp>(
        loc, viewTy, src, rewriter.getIndexAttr(viewOffset), sizes, strides);
    rewriter.replaceOpWithNewOp<memref::CopyOp>(generic, view, dst);
    return success();
  }
};

class GatherToMemrefCopy
    : public impl::GatherToMemrefCopyBase<GatherToMemrefCopy> {
public:
  using impl::GatherToMemrefCopyBase<GatherToMemrefCopy>::GatherToMemrefCopyBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<GatherToCopy>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
