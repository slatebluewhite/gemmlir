//===- FillToMemsetPass.cpp --------------------------------------*- C++ -*-===//
//
// A fill of one repeated byte is a memset.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FILLTOMEMSET
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The single byte a constant repeats, if it repeats one.
///
/// Zero always does, whatever the type, and zero is what a convolution's
/// padding and an accumulator's initialisation are. A general f32 does not:
/// 1.0 is `00 00 80 3F`, four different bytes, and there is no memset for it.
static std::optional<uint8_t> repeatedByte(Value v) {
  llvm::APInt bits;
  llvm::APFloat f(0.0f);
  if (matchPattern(v, m_ConstantInt(&bits))) {
    // A constant wider than its type is not this op's business.
  } else if (matchPattern(v, m_ConstantFloat(&f))) {
    bits = f.bitcastToAPInt();
  } else {
    return std::nullopt;
  }
  unsigned width = bits.getBitWidth();
  if (width % 8 != 0 || width > 64)
    return std::nullopt;
  uint64_t raw = bits.getZExtValue();
  uint8_t first = (uint8_t)(raw & 0xff);
  for (unsigned b = 1; b < width / 8; b++)
    if ((uint8_t)((raw >> (8 * b)) & 0xff) != first)
      return std::nullopt;
  return first;
}

/// True when the memref's elements sit one after another with nothing between.
///
/// A dimension of **size one** is skipped, whatever its stride: it never steps,
/// so it cannot leave a gap. That is not a detail --
/// `--fill-only-the-border` turns a padding's fill into slabs, and the slab
/// that is the last rows entire, `[1, 2, 14, 480]` of a `1x14x14x480` buffer,
/// is one contiguous run at the end of it. Its outermost stride is the whole
/// buffer's 94080 and its extent is 1, and insisting that 94080 equal 13440
/// refused every one of them. The slab that is the last *columns* is a run per
/// row and is still correctly refused.
static bool isContiguous(MemRefType ty) {
  if (!ty.hasStaticShape())
    return false;
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(ty.getStridesAndOffset(strides, offset)) ||
      llvm::any_of(strides, ShapedType::isDynamic))
    return false;
  int64_t packed = 1;
  for (int d = ty.getRank() - 1; d >= 0; d--) {
    if (ty.getDimSize(d) == 1)
      continue;
    if (strides[d] != packed)
      return false;
    packed *= ty.getDimSize(d);
  }
  return ty.getElementType().isIntOrFloat() &&
         ty.getElementType().getIntOrFloatBitWidth() % 8 == 0;
}

/// How many trailing dimensions of `ty` sit one after another, and how many
/// elements that run holds.
///
/// `--fill-only-the-border` cuts a padding's fill into slabs and the slab that
/// is the last *columns* of every row is a run per row, not one run: shape
/// `[1, 48, 2, 64]` over strides `[160000, 3200, 64, 1]` is 48 runs of 128
/// elements. `isContiguous` correctly refuses it, and refusing is the end of
/// it today -- so the whole slab stays a store per byte.
static std::pair<int64_t, int64_t> contiguousRun(MemRefType ty) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(ty.getStridesAndOffset(strides, offset)))
    return {0, 0};
  int64_t packed = 1, dims = 0;
  for (int d = ty.getRank() - 1; d >= 0; d--) {
    // A dimension of size one never steps, so it neither leaves a gap nor
    // extends the run.
    if (ty.getDimSize(d) == 1) {
      dims++;
      continue;
    }
    if (ShapedType::isDynamic(strides[d]) || strides[d] != packed)
      break;
    packed *= ty.getDimSize(d);
    dims++;
  }
  return {dims, packed};
}

/// A `memset` per contiguous run, under a loop over the dimensions that step
/// over the gaps.
///
/// Worth a call only when the run is long enough to pay for one: below a cache
/// line the stores the call saves do not cover the call itself.
static constexpr int64_t kLeastRunBytes = 64;

static LogicalResult memsetEachRun(PatternRewriter &rewriter, Operation *fill,
                                   Value buffer, uint8_t byte) {
  auto ty = llvm::cast<MemRefType>(buffer.getType());
  auto [runDims, runElements] = contiguousRun(ty);
  int64_t rank = ty.getRank();
  if (runDims <= 0 || runDims >= rank)
    return failure();
  int64_t width = ty.getElementType().getIntOrFloatBitWidth() / 8;
  if (width == 0 || runElements * width < kLeastRunBytes)
    return failure();

  // Everything this pattern can refuse is decided before it writes anything:
  // a pattern that fails after rewriting leaves the IR broken.
  Location loc = fill->getLoc();
  int64_t outer = rank - runDims;
  bool anyStepping = false;
  for (int64_t d = 0; d < outer; d++)
    anyStepping |= ty.getDimSize(d) != 1;
  if (!anyStepping)
    return failure();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes(rank, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> steps(rank, rewriter.getIndexAttr(1));
  for (int64_t d = outer; d < rank; d++)
    sizes[d] = rewriter.getIndexAttr(ty.getDimSize(d));

  // One loop for each outer dimension that actually steps. A leading unit
  // dimension is an index of zero, not a loop of one.
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  for (int64_t d = 0; d < outer; d++) {
    if (ty.getDimSize(d) == 1)
      continue;
    Value ub = rewriter.create<arith::ConstantIndexOp>(loc, ty.getDimSize(d));
    auto loop = rewriter.create<scf::ForOp>(loc, zero, ub, one);
    offsets[d] = loop.getInductionVar();
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  Value run = rewriter.create<memref::SubViewOp>(loc, buffer, offsets, sizes,
                                                 steps);
  rewriter.create<MemsetOp>(loc, run, rewriter.getI8IntegerAttr(byte));
  rewriter.setInsertionPoint(fill);
  rewriter.eraseOp(fill);
  return success();
}

/// The buffer a value ultimately views.
/// A `memref.view` is where the walk **stops**, not another step.
///
/// `--plan-static-buffers` puts every tensor in the model at its own offset of
/// one global arena, so walking through the view reaches that global -- and
/// then every buffer in the function has the same base and the two guards
/// below, which ask "does a later loop touch this buffer", answer yes for
/// everything. This pass converted 4 of GoogLeNet's 71 fills and nobody
/// noticed, because a pass that refuses is indistinguishable from a pass with
/// nothing to do.
///
/// Stopping at the view is exact here: the planner is a bump allocator and
/// never reuses a slot, so two views at different offsets are different
/// buffers. It is also only a performance question -- a `memset` computes the
/// same bytes as the fill either way.
///
/// With that, the size-one dimension below, the same rule in `MemsetOp`'s own
/// verifier, and `below-reduction=1`, GoogLeNet converts **29 fills of 156,176
/// elements** where it converted none: 347.66 -> **341.60** ms, -1.7%, byte for
/// byte against both references.
static Value viewedBuffer(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto op = llvm::dyn_cast<memref::SubViewOp>(def)) { v = op.getSource(); continue; }
    if (llvm::isa<memref::ViewOp>(def)) break;
    if (auto op = llvm::dyn_cast<memref::CastOp>(def)) { v = op.getSource(); continue; }
    if (auto op = llvm::dyn_cast<memref::ReinterpretCastOp>(def)) { v = op.getSource(); continue; }
    if (auto op = llvm::dyn_cast<memref::ExpandShapeOp>(def)) { v = op.getSrc(); continue; }
    if (auto op = llvm::dyn_cast<memref::CollapseShapeOp>(def)) { v = op.getSrc(); continue; }
    break;
  }
  return v;
}

/// True when a later `linalg` reduction in the same block reads or writes this
/// buffer.
///
/// Every `linalg` operation still here is a scalar loop -- the accelerator ones
/// became `gemmlir` operations before this pass runs -- and `gemmlir_memset` is
/// an opaque call, so a loop across it has to treat the buffer as
/// unknown-modified. A reduction is where that costs: its accumulator stops
/// living in a register and is reloaded every step. Measured on the board,
/// converting the fill in front of one took `shub` from 56.6 to 83.7 ms and
/// `apb` from 42.8 to 59.0 -- every model with a convolution or a pooling still
/// running as a scalar loop got worse, and every model without one got better.
///
/// An elementwise loop over the buffer is not a reason to refuse: it has no
/// accumulator to spill, and on the model set those conversions are worth
/// 11 ms.
static bool aReductionTouchesItLater(Operation *fill, Value buffer) {
  Value base = viewedBuffer(buffer);
  for (Operation *n = fill->getNextNode(); n; n = n->getNextNode()) {
    auto loop = llvm::dyn_cast<linalg::LinalgOp>(n);
    if (!loop || loop.getNumReductionLoops() == 0)
      continue;
    for (Value operand : n->getOperands())
      if (llvm::isa<MemRefType>(operand.getType()) &&
          viewedBuffer(operand) == base)
        return true;
  }
  return false;
}

/// A fill whose buffer a later loop *reads back while writing* is that loop's
/// accumulator, reduction or not, and it stays a loop for the same reason.
///
/// Writing into it without reading it is not that. A convolution's padding is
/// filled and then the real input is written into the middle of it, and that
/// write is elementwise with no accumulator to spill -- treating it as one kept
/// every padding in the grouped family a scalar store loop.
static bool accumulatedIntoLater(Operation *fill, Value buffer) {
  Value base = viewedBuffer(buffer);
  for (Operation *n = fill->getNextNode(); n; n = n->getNextNode()) {
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(n);
    if (!linalgOp)
      continue;
    for (OpOperand &init : linalgOp.getDpsInitsMutable()) {
      if (viewedBuffer(init.get()) != base)
        continue;
      // The block argument for an init the body never reads is a destination,
      // not an accumulator.
      BlockArgument arg = linalgOp.getMatchingBlockArgument(&init);
      if (!arg || !arg.use_empty())
        return true;
    }
  }
  return false;
}

/// `linalg.fill` of a uniform byte over a contiguous buffer is `memset`.
///
/// It is most of what is left of a convolution's padding: the fill covers the
/// whole padded buffer and the real input is copied into the middle of it, so
/// on the model set these are 124000 elements of scalar stores -- the largest
/// single class after the im2col packs, which are already `memcpy`s.
///
/// This has to run after `--convert-linalg-to-gemmlir`: a zero fill is how an
/// accumulator is proved to start at zero, and the conversion reads them.
class FillToMemset : public OpRewritePattern<linalg::FillOp> {
public:
  FillToMemset(MLIRContext *ctx, bool belowReduction)
      : OpRewritePattern(ctx), belowReduction(belowReduction) {}

  LogicalResult matchAndRewrite(linalg::FillOp fill,
                                PatternRewriter &rewriter) const final {
    if (!fill.hasPureBufferSemantics() || fill.getInputs().size() != 1 ||
        fill.getOutputs().size() != 1)
      return failure();
    auto ty = llvm::dyn_cast<MemRefType>(fill.getOutputs()[0].getType());
    if (!ty || !ty.hasStaticShape())
      return failure();
    std::optional<uint8_t> byte = repeatedByte(fill.getInputs()[0]);
    if (!byte)
      return failure();
    if (accumulatedIntoLater(fill, fill.getOutputs()[0]) ||
        (!belowReduction &&
         aReductionTouchesItLater(fill, fill.getOutputs()[0])))
      return failure();
    if (!isContiguous(ty))
      return memsetEachRun(rewriter, fill, fill.getOutputs()[0], *byte);
    rewriter.replaceOpWithNewOp<MemsetOp>(fill, fill.getOutputs()[0],
                                          rewriter.getI8IntegerAttr(*byte));
    return success();
  }

private:
  bool belowReduction;
};

/// Bufferizing a `tensor.pad` leaves the fill as a `linalg.map` with no inputs
/// whose body yields the constant, not as a `linalg.fill`.
class MapFillToMemset : public OpRewritePattern<linalg::MapOp> {
public:
  MapFillToMemset(MLIRContext *ctx, bool belowReduction)
      : OpRewritePattern(ctx), belowReduction(belowReduction) {}

  LogicalResult matchAndRewrite(linalg::MapOp map,
                                PatternRewriter &rewriter) const final {
    if (!map.hasPureBufferSemantics() || map.getInputs().size() != 0)
      return failure();
    auto ty = llvm::dyn_cast<MemRefType>(map.getInit().getType());
    if (!ty || !ty.hasStaticShape())
      return failure();
    Block &body = map.getMapper().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    std::optional<uint8_t> byte = repeatedByte(yield.getOperand(0));
    if (!byte)
      return failure();
    if (accumulatedIntoLater(map, map.getInit()) ||
        (!belowReduction && aReductionTouchesItLater(map, map.getInit())))
      return failure();
    if (!isContiguous(ty))
      return memsetEachRun(rewriter, map, map.getInit(), *byte);
    rewriter.replaceOpWithNewOp<MemsetOp>(map, map.getInit(),
                                          rewriter.getI8IntegerAttr(*byte));
    return success();
  }

private:
  bool belowReduction;
};

class FillToMemsetPass : public impl::FillToMemsetBase<FillToMemsetPass> {
public:
  using impl::FillToMemsetBase<FillToMemsetPass>::FillToMemsetBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, GemmlirDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FillToMemset, MapFillToMemset>(&getContext(), belowReduction);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
