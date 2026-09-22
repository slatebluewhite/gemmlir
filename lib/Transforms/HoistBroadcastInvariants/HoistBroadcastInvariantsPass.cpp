//===- HoistBroadcastInvariantsPass.cpp --------------------------*- C++ -*-===//
//
// A per-row value belongs in a per-row loop.
//
// A layer norm's inner loop multiplies a per-token reciprocal by a constant and
// then by the element -- and the first multiply is the same number for all 192
// channels, so the loop does it 192 times. Nothing else in the pipeline takes
// it, and `llc` runs no IR pipeline, so it stays.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_HOISTBROADCASTINVARIANTS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The loop dimensions a map actually names.
static llvm::SmallBitVector dimsUsed(AffineMap map, unsigned numDims) {
  llvm::SmallBitVector used(numDims);
  for (AffineExpr e : map.getResults())
    e.walk([&](AffineExpr sub) {
      if (auto d = llvm::dyn_cast<AffineDimExpr>(sub))
        used.set(d.getPosition());
    });
  return used;
}

static int64_t elementsOf(Value v) {
  auto ty = llvm::dyn_cast<MemRefType>(v.getType());
  if (!ty || !ty.hasStaticShape())
    return -1;
  int64_t n = 1;
  for (int64_t d : ty.getShape())
    n *= d;
  return n;
}

/// Worth moving out: anything but a load or a store, which a body does not
/// have, and cheap enough that a buffer is not a worse deal. Everything a
/// quantization tail is made of qualifies.
static bool movable(Operation *op) {
  return llvm::isa<arith::MulFOp, arith::DivFOp, arith::AddFOp, arith::SubFOp,
                   arith::NegFOp, arith::MaxNumFOp, arith::MinNumFOp,
                   math::RsqrtOp, math::SqrtOp, math::ExpOp, math::FmaOp,
                   arith::MulIOp, arith::AddIOp, arith::SubIOp,
                   arith::ShRSIOp, arith::ShLIOp, arith::SIToFPOp,
                   arith::FPToSIOp, arith::ExtSIOp, arith::TruncIOp,
                   arith::TruncFOp, arith::ExtFOp>(op);
}

class HoistBroadcastInvariants
    : public impl::HoistBroadcastInvariantsBase<HoistBroadcastInvariants> {
public:
  using impl::HoistBroadcastInvariantsBase<
      HoistBroadcastInvariants>::HoistBroadcastInvariantsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() final {
    SmallVector<linalg::GenericOp> work;
    getOperation().walk([&](linalg::GenericOp g) { work.push_back(g); });
    bool any = false;
    for (linalg::GenericOp g : work)
      if (g->getBlock() && succeeded(hoist(g)))
        any = true;
    (void)any;
  }

private:
  LogicalResult hoist(linalg::GenericOp generic);
};

LogicalResult HoistBroadcastInvariants::hoist(linalg::GenericOp generic) {
  if (generic.getNumDpsInits() != 1 || generic.getNumDpsInputs() < 1)
    return failure();
  for (utils::IteratorType it : generic.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel)
      return failure();
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  unsigned numDims = generic.getNumLoops();
  if (!maps.back().isIdentity())
    return failure();
  auto outTy = llvm::dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
  if (!outTy || !outTy.hasStaticShape())
    return failure();

  // Which block arguments are narrow: read through a map that misses at least
  // one loop dimension. The set of dimensions they do name has to be the same
  // for all of them, or the value would not have one shape to live in.
  // Narrow by **element count**, not by which dimensions the map names: a
  // batch of one is written as a constant zero
  // ([[mlir-unit-axis-is-a-constant-zero]]), so a full-size operand names
  // three dimensions of four and would otherwise look narrow.
  int64_t outElements = elementsOf(generic.getOutputs()[0]);
  if (outElements <= 0)
    return failure();
  llvm::SmallBitVector narrowDims(numDims);
  bool haveNarrow = false;
  llvm::SmallBitVector isNarrowArg(generic.getNumDpsInputs());
  for (unsigned i = 0; i < generic.getNumDpsInputs(); i++) {
    int64_t n = elementsOf(generic.getInputs()[i]);
    if (n <= 0 || n >= outElements)
      continue;
    llvm::SmallBitVector used = dimsUsed(maps[i], numDims);
    if (used.count() == 0 || used.count() == numDims)
      continue;
    if (!haveNarrow) {
      narrowDims = used;
      haveNarrow = true;
    } else if (narrowDims != used) {
      return failure();
    }
    isNarrowArg.set(i);
  }
  if (!haveNarrow)
    return failure();

  // The candidates: operations every operand of which is a narrow argument, a
  // constant, or another candidate.
  Block &body = generic.getRegion().front();
  llvm::SmallDenseSet<Operation *> invariant;
  SmallVector<Operation *> order;
  for (Operation &opRef : body.without_terminator()) {
    Operation *op = &opRef;
    if (!movable(op))
      continue;
    bool ok = true;
    for (Value v : op->getOperands()) {
      if (auto arg = llvm::dyn_cast<BlockArgument>(v)) {
        if (arg.getOwner() != &body ||
            arg.getArgNumber() >= generic.getNumDpsInputs() ||
            !isNarrowArg.test(arg.getArgNumber()))
          ok = false;
        continue;
      }
      Operation *def = v.getDefiningOp();
      if (!def)
        ok = false;
      else if (def->getBlock() != &body) {
        // Defined outside the region: a constant the body reads.
        if (!def->hasTrait<OpTrait::ConstantLike>())
          ok = false;
      } else if (!invariant.count(def)) {
        ok = false;
      }
    }
    if (!ok)
      continue;
    invariant.insert(op);
    order.push_back(op);
  }
  if (order.empty())
    return failure();

  // Only the values the rest of the body still needs have to come back through
  // a buffer; an operation whose every user is also invariant costs nothing.
  SmallVector<Operation *> escaping;
  for (Operation *op : order) {
    bool used = false;
    for (Operation *user : op->getUsers())
      if (!invariant.count(user))
        used = true;
    if (used)
      escaping.push_back(op);
  }
  // One buffer per escaping value; more than a couple and the loop is better
  // off recomputing.
  if (escaping.empty() || escaping.size() > 2)
    return failure();
  for (Operation *op : escaping)
    if (op->getNumResults() != 1)
      return failure();

  // The narrow shape, taken from one of the narrow operands: they all name the
  // same dimensions, so they all have it.
  Value model;
  for (unsigned i = 0; i < generic.getNumDpsInputs(); i++)
    if (isNarrowArg.test(i)) {
      model = generic.getInputs()[i];
      break;
    }
  auto modelTy = llvm::cast<MemRefType>(model.getType());
  if (!modelTy.hasStaticShape())
    return failure();
  int64_t narrowEls = elementsOf(model), outEls = elementsOf(generic.getOutputs()[0]);
  if (narrowEls <= 0 || outEls <= 0 || narrowEls >= outEls)
    return failure();

  // Build the small loop: the same narrow operands, the invariant operations,
  // one output per escaping value.
  OpBuilder b(generic);
  Location loc = generic.getLoc();
  SmallVector<Value> smallIns;
  SmallVector<unsigned> smallArgOf(generic.getNumDpsInputs(), 0);
  SmallVector<AffineMap> smallMaps;
  unsigned narrowRank = modelTy.getRank();
  AffineMap ident = AffineMap::getMultiDimIdentityMap(narrowRank, b.getContext());
  for (unsigned i = 0; i < generic.getNumDpsInputs(); i++)
    if (isNarrowArg.test(i)) {
      auto ty = llvm::dyn_cast<MemRefType>(generic.getInputs()[i].getType());
      if (!ty || ty.getRank() != (int64_t)narrowRank ||
          ty.getShape() != modelTy.getShape())
        return failure();
      smallArgOf[i] = smallIns.size();
      smallIns.push_back(generic.getInputs()[i]);
      smallMaps.push_back(ident);
    }

  SmallVector<Value> smallOuts;
  for (Operation *op : escaping) {
    auto ty = MemRefType::get(modelTy.getShape(), op->getResult(0).getType());
    smallOuts.push_back(b.create<memref::AllocOp>(loc, ty, b.getI64IntegerAttr(64)));
    smallMaps.push_back(ident);
  }
  SmallVector<utils::IteratorType> smallIters(narrowRank,
                                              utils::IteratorType::parallel);

  SmallVector<Operation *> orderedInvariant = order;
  b.create<linalg::GenericOp>(
      loc, TypeRange{}, smallIns, smallOuts, smallMaps, smallIters,
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        IRMapping map;
        for (unsigned i = 0; i < generic.getNumDpsInputs(); i++)
          if (isNarrowArg.test(i))
            map.map(body.getArgument(i), args[smallArgOf[i]]);
        for (Operation *op : orderedInvariant)
          nb.clone(*op, map);
        SmallVector<Value> yields;
        for (Operation *op : escaping)
          yields.push_back(map.lookup(op->getResult(0)));
        nb.create<linalg::YieldOp>(nl, yields);
      });

  // The element loop reads the results instead of recomputing them.
  SmallVector<Value> newIns(generic.getInputs().begin(),
                            generic.getInputs().end());
  SmallVector<AffineMap> newMaps = maps;
  AffineMap narrowMap;
  for (unsigned i = 0; i < generic.getNumDpsInputs(); i++)
    if (isNarrowArg.test(i)) {
      narrowMap = maps[i];
      break;
    }
  for (Value v : smallOuts) {
    newIns.push_back(v);
    newMaps.insert(newMaps.begin() + newIns.size() - 1, narrowMap);
  }
  auto replacement = b.create<linalg::GenericOp>(
      loc, TypeRange{}, newIns, generic.getOutputs(), newMaps,
      llvm::to_vector(generic.getIteratorTypesArray()),
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        IRMapping map;
        for (unsigned i = 0; i < generic.getNumDpsInputs(); i++)
          map.map(body.getArgument(i), args[i]);
        map.map(body.getArguments().back(), args.back());
        for (unsigned k = 0; k < escaping.size(); k++)
          map.map(escaping[k]->getResult(0),
                  args[generic.getNumDpsInputs() + k]);
        for (Operation &opRef : body.without_terminator()) {
          if (invariant.count(&opRef))
            continue;
          nb.clone(opRef, map);
        }
        Operation *yield = body.getTerminator();
        SmallVector<Value> outs;
        for (Value v : yield->getOperands())
          outs.push_back(map.lookup(v));
        nb.create<linalg::YieldOp>(nl, outs);
      });
  (void)replacement;

  OpBuilder fb(generic);
  fb.setInsertionPointAfter(generic);
  for (Value v : smallOuts)
    fb.create<memref::DeallocOp>(loc, v);
  generic.erase();
  return success();
}

} // namespace

} // namespace mlir::gemmlir
