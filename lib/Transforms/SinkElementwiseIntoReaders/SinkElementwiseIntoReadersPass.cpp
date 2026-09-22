//===- SinkElementwiseIntoReadersPass.cpp ------------------------*- C++ -*-===//
//
// A buffer read twice is a buffer written once too often.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_SINKELEMENTWISEINTOREADERS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The buffer a value names, with every reindexing walked off. Two values can
/// only touch the same bytes if this is the same `Value` -- or the same global.
static Value baseBuffer(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto sub = llvm::dyn_cast<memref::SubViewOp>(def)) { v = sub.getSource(); continue; }
    if (auto ex = llvm::dyn_cast<memref::ExpandShapeOp>(def)) { v = ex.getSrc(); continue; }
    if (auto co = llvm::dyn_cast<memref::CollapseShapeOp>(def)) { v = co.getSrc(); continue; }
    if (auto ca = llvm::dyn_cast<memref::CastOp>(def)) { v = ca.getSource(); continue; }
    if (auto vw = llvm::dyn_cast<memref::ViewOp>(def)) { v = vw.getSource(); continue; }
    break;
  }
  return v;
}

/// Distinctness has to be *proved*, not assumed. Two `memref.alloc`s are
/// different buffers, two globals are the same buffer exactly when they are the
/// same symbol, and anything else -- a block argument, an unknown producer --
/// counts as possibly the same.
static bool provablyDistinct(Value a, Value b) {
  a = baseBuffer(a);
  b = baseBuffer(b);
  if (a == b)
    return false;
  auto kind = [](Value v) -> int {
    Operation *def = v.getDefiningOp();
    if (llvm::isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(def))
      return 1;
    if (llvm::isa_and_nonnull<memref::GetGlobalOp>(def))
      return 2;
    return 0;
  };
  int ka = kind(a), kb = kind(b);
  if (!ka || !kb)
    return false;
  if (ka != kb)
    return true;
  if (ka == 1)
    return true; // two different allocations
  return llvm::cast<memref::GetGlobalOp>(a.getDefiningOp()).getName() !=
         llvm::cast<memref::GetGlobalOp>(b.getDefiningOp()).getName();
}

/// The same type over a different element: a `strided` layout counts in
/// elements, so it carries across unchanged.
static MemRefType withElementType(MemRefType ty, Type elt) {
  return MemRefType::get(ty.getShape(), elt, ty.getLayout(),
                         ty.getMemorySpace());
}

static bool isViewOp(Operation *op) {
  return llvm::isa_and_nonnull<memref::SubViewOp, memref::ExpandShapeOp,
                               memref::CollapseShapeOp, memref::CastOp>(op);
}

/// Rebuild one reindexing on a different buffer of the same shape. Only the
/// element type moves; every offset, size, stride and reassociation is the one
/// the original carried.
static Value replayView(OpBuilder &b, Operation *view, Value src) {
  Location loc = view->getLoc();
  auto srcTy = llvm::cast<MemRefType>(src.getType());
  auto resTy = withElementType(llvm::cast<MemRefType>(view->getResult(0).getType()),
                               srcTy.getElementType());
  if (auto sub = llvm::dyn_cast<memref::SubViewOp>(view))
    return b.create<memref::SubViewOp>(loc, resTy, src, sub.getMixedOffsets(),
                                       sub.getMixedSizes(), sub.getMixedStrides());
  if (auto ex = llvm::dyn_cast<memref::ExpandShapeOp>(view))
    return b.create<memref::ExpandShapeOp>(loc, resTy, src,
                                           ex.getReassociationIndices(),
                                           ex.getMixedOutputShape());
  if (auto co = llvm::dyn_cast<memref::CollapseShapeOp>(view))
    return b.create<memref::CollapseShapeOp>(loc, resTy, src,
                                             co.getReassociationIndices());
  return b.create<memref::CastOp>(loc, resTy, src);
}

static int64_t bufferBytes(Value v) {
  auto ty = llvm::dyn_cast<MemRefType>(v.getType());
  if (!ty || !ty.hasStaticShape())
    return -1;
  unsigned bits = ty.getElementType().getIntOrFloatBitWidth();
  return ty.getNumElements() * ((bits + 7) / 8);
}

/// An elementwise `linalg.generic` on buffers whose result is read by a handful
/// of other loops and by nobody else. Recompute it in each of them and the
/// buffer -- the write *and* every read of it -- goes away.
///
/// A ViT's layer norm is the shape this was built for. It writes `x - mean`
/// into a 17x192 f32 buffer, 13 KB against a 16 KB L1, and reads it twice: once
/// for the variance and once to normalize. Upstream elementwise fusion will not
/// touch it because fusing needs `hasOneUse` and this has two. Measured as a
/// kernel on the board, recomputing the subtraction in both readers is
/// **-12.8%** on the layer norm, and byte for byte the same answer -- it is the
/// same arithmetic in the same order, only not stored in between.
///
/// **Recomputing is not always cheaper, and the bound is traffic, not count.**
/// DenseNet's dequantize feeds the batch norm of every later layer in the block
/// -- up to 25 readers -- and doing it 25 times instead of once measured
/// **+0.9%** on the board. So the rule is the one that follows from counting
/// bytes:
///
/// ```
///   before = sum(inputs) + write(B) + N * read(B)
///   after  = N * sum(inputs)
/// ```
///
/// which is worth it exactly when `(N-1) * sum(inputs) < (N+1) * size(B)`, plus
/// a hard cap on `N` and on the size of the body, because the arithmetic the
/// recomputation repeats is not in that inequality.
class SinkElementwiseIntoReaders : public OpRewritePattern<linalg::GenericOp> {
public:
  SinkElementwiseIntoReaders(MLIRContext *ctx, int64_t maxConsumers,
                             int64_t maxOps)
      : OpRewritePattern(ctx), maxConsumers(maxConsumers), maxOps(maxOps) {}

  LogicalResult matchAndRewrite(linalg::GenericOp producer,
                                PatternRewriter &rewriter) const final {
    if (producer->getNumResults() != 0 || producer.getOutputs().size() != 1 ||
        producer.getInputs().empty())
      return failure();
    for (utils::IteratorType it : producer.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();

    SmallVector<AffineMap> pMaps = producer.getIndexingMapsArray();
    // The output map has to be the identity, so that an index into the buffer
    // *is* an index into the producer's iteration space and the consumer's map
    // can simply be composed with the input maps.
    if (!pMaps.back().isIdentity())
      return failure();

    Value buffer = producer.getOutputs()[0];
    auto bufTy = llvm::dyn_cast<MemRefType>(buffer.getType());
    if (!bufTy || !bufTy.hasStaticShape() || !bufTy.getLayout().isIdentity())
      return failure();
    auto alloc = buffer.getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    Block &pBody = producer.getRegion().front();
    unsigned pNumIn = producer.getInputs().size();
    // The `outs` argument is whatever the buffer already held; a fresh
    // allocation holds nothing, and recomputing could not reproduce it anyway.
    for (unsigned i = pNumIn; i < pBody.getNumArguments(); i++)
      if (!pBody.getArgument(i).use_empty())
        return failure();
    int64_t bodyOps = 0;
    for (Operation &op : pBody.without_terminator()) {
      if (!isMemoryEffectFree(&op) || op.getNumRegions() || op.getNumResults() != 1)
        return failure();
      // `linalg.index` reads the loop it is in, and the consumer's loop is a
      // different one.
      if (llvm::isa<linalg::IndexOp>(op))
        return failure();
      if (++bodyOps > maxOps)
        return failure();
    }
    auto pYield = llvm::cast<linalg::YieldOp>(pBody.getTerminator());
    if (pYield.getNumOperands() != 1)
      return failure();

    // Collect the readers, following the buffer through any reindexing taken of
    // it. Everything else touching it -- a second writer, an escape -- stops
    // the rewrite.
    SmallVector<linalg::GenericOp> consumers;
    SmallVector<unsigned> operandIdx;
    SmallVector<Value> readValue;   // the buffer, or a view of it
    SmallVector<Operation *> deallocs;
    SmallVector<Operation *> views;
    SmallVector<Value> worklist{buffer};
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (user == producer)
          continue;
        if (llvm::isa<memref::DeallocOp>(user)) {
          deallocs.push_back(user);
          continue;
        }
        if (isViewOp(user)) {
          if (use.getOperandNumber() != 0)
            return failure();
          views.push_back(user);
          worklist.push_back(user->getResult(0));
          continue;
        }
        auto consumer = llvm::dyn_cast<linalg::GenericOp>(user);
        if (!consumer || consumer->getBlock() != producer->getBlock())
          return failure();
        unsigned idx = use.getOperandNumber();
        if (idx >= consumer.getInputs().size())
          return failure(); // written, not read
        if (llvm::is_contained(consumers, consumer))
          return failure(); // read twice by one loop
        consumers.push_back(consumer);
        operandIdx.push_back(idx);
        readValue.push_back(v);
      }
    }
    if (consumers.empty() || (int64_t)consumers.size() > maxConsumers)
      return failure();

    // A reader that goes through a reindexing is served by replaying that same
    // reindexing on the producer's inputs, which only lines up when every
    // operand has the buffer's shape and the identity map.
    bool viewed = !views.empty();
    if (viewed)
      for (auto [map, operand] :
           llvm::zip(pMaps, producer->getOperands()))
        if (!map.isIdentity() ||
            llvm::cast<MemRefType>(operand.getType()).getShape() !=
                bufTy.getShape())
          return failure();

    // Bytes, not operations: see the comment above the class. A reader that
    // takes a slice reads only that slice, which is what makes a dequantize
    // split three ways -- a transformer's Q, K and V -- free rather than triple.
    int64_t bufBytes = bufferBytes(buffer), inBytes = 0, readBytes = 0;
    if (bufBytes <= 0)
      return failure();
    for (Value in : producer.getInputs()) {
      int64_t b = bufferBytes(in);
      if (b < 0)
        return failure();
      inBytes += b;
    }
    for (Value v : readValue) {
      int64_t b = bufferBytes(v);
      if (b < 0)
        return failure();
      readBytes += b;
    }
    // before = inBytes + write(buffer) + readBytes
    // after  = readBytes * inBytes / bufBytes
    if (readBytes * inBytes >= (inBytes + bufBytes + readBytes) * bufBytes)
      return failure();

    // Recomputing reads the producer's inputs where the consumer stands, so
    // nothing may have written them in between.
    for (linalg::GenericOp consumer : consumers)
      if (!inputsSurvive(producer, consumer))
        return failure();

    for (auto [consumer, idx, v] : llvm::zip(consumers, operandIdx, readValue))
      rewrite(producer, consumer, idx, v, buffer, rewriter);

    rewriter.eraseOp(producer);
    for (Operation *d : deallocs)
      rewriter.eraseOp(d);
    // The reindexings were uses of the buffer and have none of their own left.
    for (Operation *view : llvm::reverse(views))
      rewriter.eraseOp(view);
    rewriter.eraseOp(alloc);
    return success();
  }

private:
  /// Whether `op`, or anything nested inside it, may write one of `inputs`.
  ///
  /// Two things here are not the obvious code. `scf.for` **declares no memory
  /// effects at all** -- it carries `RecursiveMemoryEffects` and no interface of
  /// its own -- so asking it directly says "unknown" and refuses every
  /// transformer, which is what [[gemmlir-scf-for-declares-no-effects]] cost
  /// once already; the walk reaches its body instead. And a `gemmlir` operation
  /// declares a write that is **tied to no operand**, so every buffer it holds
  /// has to be treated as the target.
  bool mayWriteAny(Operation *op, ValueRange inputs) const {
    if (isMemoryEffectFree(op))
      return false;
    bool bad = false;
    op->walk([&](Operation *n) {
      if (isMemoryEffectFree(n))
        return WalkResult::advance();
      auto iface = llvm::dyn_cast<MemoryEffectOpInterface>(n);
      if (!iface) {
        if (n->getNumRegions())
          return WalkResult::advance(); // it is exactly what it contains
        bad = true;
        return WalkResult::interrupt();
      }
      SmallVector<MemoryEffects::EffectInstance> found;
      iface.getEffects(found);
      auto clashes = [&](Value written) {
        for (Value in : inputs)
          if (!provablyDistinct(written, in))
            return true;
        return false;
      };
      for (const MemoryEffects::EffectInstance &e : found) {
        if (llvm::isa<MemoryEffects::Read, MemoryEffects::Allocate>(e.getEffect()))
          continue;
        if (Value written = e.getValue()) {
          if (clashes(written)) {
            bad = true;
            return WalkResult::interrupt();
          }
          continue;
        }
        for (Value operand : n->getOperands())
          if (llvm::isa<MemRefType>(operand.getType()) && clashes(operand)) {
            bad = true;
            return WalkResult::interrupt();
          }
      }
      return WalkResult::advance();
    });
    return bad;
  }

  /// Nothing between the producer and the consumer may write any buffer the
  /// producer reads.
  bool inputsSurvive(linalg::GenericOp producer,
                     linalg::GenericOp consumer) const {
    for (Operation *op = producer->getNextNode(); op && op != consumer;
         op = op->getNextNode())
      if (mayWriteAny(op, producer.getInputs()))
        return false;
    return true;
  }

  /// Replace the consumer's read of the buffer with the producer's body.
  ///
  /// `read` is what the consumer actually names: the buffer itself, or a view
  /// of it. A view is replayed on each of the producer's inputs, and because
  /// the view has the buffer's shape the consumer's own map is unchanged.
  void rewrite(linalg::GenericOp producer, linalg::GenericOp consumer,
               unsigned idx, Value read, Value buffer,
               PatternRewriter &rewriter) const {
    Location loc = consumer.getLoc();
    SmallVector<AffineMap> pMaps = producer.getIndexingMapsArray();
    SmallVector<AffineMap> cMaps = consumer.getIndexingMapsArray();
    AffineMap through = cMaps[idx]; // consumer loops -> buffer index

    // The reindexings between the buffer and what the consumer names, outermost
    // last.
    SmallVector<Operation *> chain;
    for (Value v = read; v != buffer; v = v.getDefiningOp()->getOperand(0))
      chain.push_back(v.getDefiningOp());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(consumer);

    SmallVector<Value> newIns;
    SmallVector<AffineMap> newMaps;
    unsigned pNumIn = producer.getInputs().size();
    for (unsigned j = 0; j < consumer.getInputs().size(); j++) {
      if (j != idx) {
        newIns.push_back(consumer.getInputs()[j]);
        newMaps.push_back(cMaps[j]);
        continue;
      }
      for (unsigned i = 0; i < pNumIn; i++) {
        Value in = producer.getInputs()[i];
        for (Operation *view : llvm::reverse(chain))
          in = replayView(rewriter, view, in);
        newIns.push_back(in);
        newMaps.push_back(chain.empty() ? pMaps[i].compose(through) : through);
      }
    }
    for (unsigned j = 0; j < consumer.getOutputs().size(); j++)
      newMaps.push_back(cMaps[consumer.getInputs().size() + j]);

    auto fused = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{}, newIns, consumer.getOutputs(), newMaps,
        consumer.getIteratorTypesArray());

    Block &pBody = producer.getRegion().front();
    Block &cBody = consumer.getRegion().front();
    SmallVector<Type> argTypes;
    SmallVector<Location> argLocs;
    for (Value v : newIns) {
      argTypes.push_back(llvm::cast<MemRefType>(v.getType()).getElementType());
      argLocs.push_back(loc);
    }
    for (unsigned j = consumer.getInputs().size(); j < cBody.getNumArguments();
         j++) {
      argTypes.push_back(cBody.getArgument(j).getType());
      argLocs.push_back(cBody.getArgument(j).getLoc());
    }
    Block *body = rewriter.createBlock(&fused.getRegion(),
                                       fused.getRegion().begin(), argTypes,
                                       argLocs);
    rewriter.setInsertionPointToStart(body);

    // The producer's body first, reading the arguments that took the buffer's
    // place; then the consumer's, with its buffer argument standing for what
    // the producer yields.
    IRMapping map;
    for (unsigned i = 0; i < pNumIn; i++)
      map.map(pBody.getArgument(i), body->getArgument(idx + i));
    for (Operation &op : pBody.without_terminator())
      rewriter.clone(op, map);
    Value recomputed =
        map.lookupOrDefault(llvm::cast<linalg::YieldOp>(pBody.getTerminator())
                                .getOperand(0));

    IRMapping cmap;
    for (unsigned j = 0; j < cBody.getNumArguments(); j++) {
      if (j == idx) {
        cmap.map(cBody.getArgument(j), recomputed);
      } else {
        unsigned shifted = j < idx ? j : j + pNumIn - 1;
        cmap.map(cBody.getArgument(j), body->getArgument(shifted));
      }
    }
    for (Operation &op : cBody)
      rewriter.clone(op, cmap);

    rewriter.eraseOp(consumer);
  }

  int64_t maxConsumers;
  int64_t maxOps;
};

class SinkElementwiseIntoReaders_Pass
    : public impl::SinkElementwiseIntoReadersBase<
          SinkElementwiseIntoReaders_Pass> {
public:
  using impl::SinkElementwiseIntoReadersBase<
      SinkElementwiseIntoReaders_Pass>::SinkElementwiseIntoReadersBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, arith::ArithDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SinkElementwiseIntoReaders>(&getContext(), maxConsumers,
                                             maxOps);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
