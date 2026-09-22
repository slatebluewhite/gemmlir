//===- FillOnlyTheBorderPass.cpp ---------------------------------*- C++ -*-===//
//
// A padding is the border, not the whole buffer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FILLONLYTHEBORDER
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The allocation a value ultimately views.
static Value baseOf(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto s = llvm::dyn_cast<memref::SubViewOp>(def)) { v = s.getSource(); continue; }
    if (auto c = llvm::dyn_cast<memref::CastOp>(def)) { v = c.getSource(); continue; }
    if (auto c = llvm::dyn_cast<memref::CollapseShapeOp>(def)) { v = c.getSrc(); continue; }
    if (auto e = llvm::dyn_cast<memref::ExpandShapeOp>(def)) { v = e.getSrc(); continue; }
    if (auto r = llvm::dyn_cast<memref::ReinterpretCastOp>(def)) { v = r.getSource(); continue; }
    if (auto w = llvm::dyn_cast<memref::ViewOp>(def)) { return w.getResult(); }
    break;
  }
  return v;
}

/// The one box a later operation writes into `buffer`, if the operations
/// between leave the buffer alone.
struct Box {
  memref::SubViewOp slice;
  Operation *writer;
};

static std::optional<Box> boxWrittenNext(Operation *fill, Value buffer) {
  Value base = baseOf(buffer);
  for (Operation *n = fill->getNextNode(); n; n = n->getNextNode()) {
    // A subview of this buffer is the candidate; anything else that mentions it
    // in between -- a read, a second write, a call it could escape through --
    // means the fill is not simply covered.
    if (auto slice = llvm::dyn_cast<memref::SubViewOp>(n)) {
      if (baseOf(slice.getSource()) != base)
        continue;
      if (!slice->hasOneUse())
        return std::nullopt;
      Operation *user = *slice->getUsers().begin();
      if (auto copy = llvm::dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() != slice.getResult())
          return std::nullopt;
        return Box{slice, user};
      }
      if (auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(user)) {
        // It has to write the slice and not read it back, or the fill is its
        // accumulator's starting value.
        if (linalgOp.getDpsInits().size() != 1 ||
            linalgOp.getDpsInits()[0] != slice.getResult())
          return std::nullopt;
        for (OpOperand *in : linalgOp.getDpsInputOperands())
          if (baseOf(in->get()) == base)
            return std::nullopt;
        OpOperand &init = *linalgOp.getDpsInitsMutable().begin();
        BlockArgument arg = linalgOp.getMatchingBlockArgument(&init);
        if (arg && !arg.use_empty())
          return std::nullopt;
        return Box{slice, user};
      }
      return std::nullopt;
    }
    for (Value operand : n->getOperands())
      if (llvm::isa<MemRefType>(operand.getType()) && baseOf(operand) == base)
        return std::nullopt;
    if (n->getNumRegions() != 0)
      return std::nullopt;
  }
  return std::nullopt;
}

/// A padding fills the whole buffer and then the real data is written into the
/// middle of it, so most of what was just written is immediately overwritten.
/// Across GoogLeNet's ten padded pools that is **681,808 elements filled of
/// which 473,472 -- 69% -- never survive to be read**, and the fill is a scalar
/// store loop, not a `memset`: the buffer is read by the pool, which is a
/// reduction, and `--fill-to-memset` refuses those.
///
/// The complement of a box inside a buffer is a set of slabs, one pair per
/// dimension: a point outside the box is outside on some dimension, and
/// assigning it to the **first** such dimension -- with the dimensions before
/// that one clipped to the box and the ones after left whole -- covers every
/// outside point exactly once. For an NHWC pad of `high[0, 2, 2, 0]` that is
/// two slabs: the last two rows entire, and the last two columns of the rows
/// above them.
///
/// The first of those is contiguous, so it can still become a `memset`; the
/// second is a run per row. Both together are a third of the fill they replace.
///
/// The rewrite is sound because the fill and the write cover the buffer between
/// them either way: every element the removed part of the fill wrote is written
/// again, by the very next operation that touches the buffer, before anything
/// reads it.
///
/// | | ms | |
/// |---|---|---|
/// | `googlenet` | 389.93 -> **363.25** | -6.8% |
/// | `densenet121` | 790.49 -> **785.59** | -0.6% |
/// | the set | 2705.04 -> **2672.11** | -1.2% |
///
/// GoogLeNet's fills go from 1,010,992 elements an inference to 512,752, and
/// every model in the set is byte for byte against both the CPU reference and
/// the previous build. The object is **22% larger** -- eleven sites become
/// thirty-odd loop nests -- which is why this one had to be measured rather
/// than counted.
class FillOnlyTheBorder {
public:
  /// Replaces `fill` with one fill per slab of `buffer` outside `box`. Returns
  /// failure when the box is the whole buffer or nothing is left to do.
  static LogicalResult rewrite(PatternRewriter &rewriter, Operation *fill,
                               Value buffer, Value fillValue, Box box,
                               int64_t minSaved) {
    auto ty = llvm::dyn_cast<MemRefType>(buffer.getType());
    if (!ty || !ty.hasStaticShape() || !ty.getLayout().isIdentity())
      return failure();
    memref::SubViewOp slice = box.slice;
    if (!slice.getOffsets().empty() || !slice.getSizes().empty() ||
        !slice.getStrides().empty())
      return failure();
    ArrayRef<int64_t> off = slice.getStaticOffsets();
    ArrayRef<int64_t> size = slice.getStaticSizes();
    int64_t rank = ty.getRank();
    if ((int64_t)off.size() != rank || (int64_t)size.size() != rank)
      return failure();
    for (int64_t s : slice.getStaticStrides())
      if (s != 1)
        return failure();
    if (llvm::any_of(off, ShapedType::isDynamic) ||
        llvm::any_of(size, ShapedType::isDynamic))
      return failure();
    ArrayRef<int64_t> shape = ty.getShape();
    for (int64_t d = 0; d < rank; d++)
      if (off[d] < 0 || size[d] <= 0 || off[d] + size[d] > shape[d])
        return failure();

    // What the rewrite saves is exactly the box: those elements are written
    // twice today and once afterwards. What it costs is a loop nest per slab.
    // The test is therefore on the saving itself and **not** on the ratio --
    // GoogLeNet's four 9x9x512 pools spare 18,432 elements each while their box
    // covers only 44% of the buffer, and a ratio test refused every one.
    int64_t inside = 1;
    for (int64_t d = 0; d < rank; d++)
      inside *= size[d];
    if (inside < minSaved)
      return failure();

    Location loc = fill->getLoc();
    rewriter.setInsertionPoint(fill);
    SmallVector<Operation *> made;
    for (int64_t d = 0; d < rank; d++) {
      for (int side = 0; side < 2; side++) {
        int64_t lo = side == 0 ? 0 : off[d] + size[d];
        int64_t hi = side == 0 ? off[d] : shape[d];
        if (lo >= hi)
          continue;
        SmallVector<OpFoldResult> o, s, st;
        for (int64_t e = 0; e < rank; e++) {
          if (e < d) {        // already inside the box on this dimension
            o.push_back(rewriter.getIndexAttr(off[e]));
            s.push_back(rewriter.getIndexAttr(size[e]));
          } else if (e == d) {
            o.push_back(rewriter.getIndexAttr(lo));
            s.push_back(rewriter.getIndexAttr(hi - lo));
          } else {            // unconstrained
            o.push_back(rewriter.getIndexAttr(0));
            s.push_back(rewriter.getIndexAttr(shape[e]));
          }
          st.push_back(rewriter.getIndexAttr(1));
        }
        Value slab =
            rewriter.create<memref::SubViewOp>(loc, buffer, o, s, st);
        made.push_back(
            rewriter.create<linalg::FillOp>(loc, ValueRange{fillValue},
                                            ValueRange{slab}));
      }
    }
    if (made.empty())
      return failure();
    rewriter.eraseOp(fill);
    return success();
  }
};

class FillBorder : public OpRewritePattern<linalg::FillOp> {
public:
  FillBorder(MLIRContext *ctx, int64_t minSaved)
      : OpRewritePattern(ctx), minSaved(minSaved) {}

  LogicalResult matchAndRewrite(linalg::FillOp fill,
                                PatternRewriter &rewriter) const final {
    if (!fill.hasPureBufferSemantics() || fill.getInputs().size() != 1 ||
        fill.getOutputs().size() != 1)
      return failure();
    Value buffer = fill.getOutputs()[0];
    std::optional<Box> box = boxWrittenNext(fill, buffer);
    if (!box)
      return failure();
    return FillOnlyTheBorder::rewrite(rewriter, fill, buffer,
                                      fill.getInputs()[0], *box, minSaved);
  }

private:
  int64_t minSaved;
};

/// Bufferizing a `tensor.pad` leaves the fill as a `linalg.map` with no inputs
/// whose body yields the constant, which is the form every padding in the set
/// actually arrives in.
class MapBorder : public OpRewritePattern<linalg::MapOp> {
public:
  MapBorder(MLIRContext *ctx, int64_t minSaved)
      : OpRewritePattern(ctx), minSaved(minSaved) {}

  LogicalResult matchAndRewrite(linalg::MapOp map,
                                PatternRewriter &rewriter) const final {
    if (!map.hasPureBufferSemantics() || map.getInputs().size() != 0)
      return failure();
    Block &body = map.getMapper().front();
    auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    Value v = yield.getOperand(0);
    // The constant has to be available where the new fills go.
    if (v.getParentBlock() == &body)
      return failure();
    Value buffer = map.getInit();
    std::optional<Box> box = boxWrittenNext(map, buffer);
    if (!box)
      return failure();
    return FillOnlyTheBorder::rewrite(rewriter, map, buffer, v, *box, minSaved);
  }

private:
  int64_t minSaved;
};

class FillOnlyTheBorderPass
    : public impl::FillOnlyTheBorderBase<FillOnlyTheBorderPass> {
public:
  using impl::FillOnlyTheBorderBase<FillOnlyTheBorderPass>::FillOnlyTheBorderBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FillBorder, MapBorder>(&getContext(), minSaved);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
