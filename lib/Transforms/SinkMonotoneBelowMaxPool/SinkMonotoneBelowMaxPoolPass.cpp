//===- SinkMonotoneBelowMaxPoolPass.cpp --------------------------*- C++ -*-===//
//
// max(f(a), f(b)) is f(max(a, b)): put f on the smaller side.
//
// DenseNet's stem dequantizes a 32x32x64 accumulator into f32 with a
// per-channel bias and a relu, pads it, max-pools it to 16x16x64 and hands that
// on. The dequantization runs on four times as many elements as anything
// downstream reads, through a quarter-megabyte buffer that exists only to be
// copied into the padded one.
//
// A monotone non-decreasing map commutes with a maximum, so the pool can read
// the accumulator itself and the dequantization can run once on the pooled
// result. Measured as the stem's two loops on the board: **-33.6%**, and every
// one of the 16,384 outputs identical -- the same f32 arithmetic on the same
// number, because a strictly increasing map does not move which element wins.
//
// The cost is that the pool's maximum moves from `fmax.s`, one instruction, to
// a signed integer maximum, which with no Zbb is a data-dependent branch. That
// is what this was rejected for on paper, and the paper was wrong: a branch on
// this core is much cheaper than a quarter of the elementwise work
// ([[gemmlir-two-clamps-are-a-range-check]]).
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

#define GEN_PASS_DEF_SINKMONOTONEBELOWMAXPOOL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

static bool floatConst(Value v, APFloat &out) {
  Attribute attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return false;
  if (auto f = llvm::dyn_cast<FloatAttr>(attr)) {
    out = f.getValue();
    return true;
  }
  if (auto d = llvm::dyn_cast<SplatElementsAttr>(attr))
    if (auto f = llvm::dyn_cast<FloatAttr>(d.getSplatValue<Attribute>())) {
      out = f.getValue();
      return true;
    }
  return false;
}

/// `f32 = relu(sitofp(i32) * k + bias)`, and the floor the relu clamps at.
///
/// Monotone non-decreasing in the accumulator, which is what lets it move
/// across the maximum, and ending at a floor, which is what makes the smallest
/// representable integer a safe pad value: it comes out of the map at exactly
/// the number the f32 pad held.
static bool monotoneWithFloor(linalg::GenericOp map, APFloat &floor) {
  if (map.getNumDpsInits() != 1 || map.getNumDpsInputs() < 1)
    return false;
  Value cur = map.getBody()->getArgument(0);
  bool sawConvert = false, sawFloor = false;
  for (Operation &opRef : map.getBody()->without_terminator()) {
    Operation *op = &opRef;
    APFloat k(0.0f);
    if (auto c = llvm::dyn_cast<arith::SIToFPOp>(op)) {
      if (sawConvert || c.getIn() != cur)
        return false;
      sawConvert = true;
      cur = c.getResult();
    } else if (auto m = llvm::dyn_cast<arith::MulFOp>(op)) {
      // A positive constant scale; anything else is not monotone increasing.
      if (!sawConvert || sawFloor || m.getLhs() != cur ||
          !floatConst(m.getRhs(), k) || !k.isFiniteNonZero() || k.isNegative())
        return false;
      cur = m.getResult();
    } else if (auto f = llvm::dyn_cast<math::FmaOp>(op)) {
      if (!sawConvert || sawFloor || f.getA() != cur ||
          !floatConst(f.getB(), k) || !k.isFiniteNonZero() || k.isNegative())
        return false;
      cur = f.getResult();
    } else if (auto a = llvm::dyn_cast<arith::AddFOp>(op)) {
      // The offset may be a per-channel operand; adding is monotone whatever
      // it is.
      if (!sawConvert || sawFloor || a.getLhs() != cur)
        return false;
      cur = a.getResult();
    } else if (auto x = llvm::dyn_cast<arith::MaxNumFOp>(op)) {
      if (sawFloor || x.getLhs() != cur || !floatConst(x.getRhs(), k))
        return false;
      floor = k;
      sawFloor = true;
      cur = x.getResult();
    } else if (auto cmp = llvm::dyn_cast<arith::CmpFOp>(op)) {
      // The relu before `--select-to-minmax` gets to it: `cmpf ugt` then
      // `select`.
      if (sawFloor || cmp.getLhs() != cur ||
          cmp.getPredicate() != arith::CmpFPredicate::UGT ||
          !floatConst(cmp.getRhs(), k))
        return false;
      floor = k;
      continue;
    } else if (auto sel = llvm::dyn_cast<arith::SelectOp>(op)) {
      auto cmp = sel.getCondition().getDefiningOp<arith::CmpFOp>();
      APFloat k2(0.0f);
      if (!cmp || sel.getTrueValue() != cur ||
          !floatConst(sel.getFalseValue(), k2) ||
          k2.compare(floor) != APFloat::cmpEqual)
        return false;
      sawFloor = true;
      cur = sel.getResult();
    } else {
      return false;
    }
  }
  return sawConvert && sawFloor &&
         map.getBody()->getTerminator()->getOperand(0) == cur;
}

/// The one `linalg.generic` that writes `v`, when `skip` and a deallocation are
/// the only other things that touch it.
static linalg::GenericOp soleWriter(Value v, Operation *skip) {
  linalg::GenericOp found;
  for (Operation *user : v.getUsers()) {
    if (user == skip || llvm::isa<memref::DeallocOp>(user))
      continue;
    auto g = llvm::dyn_cast<linalg::GenericOp>(user);
    if (!g || found || !llvm::is_contained(g.getOutputs(), v))
      return nullptr;
    found = g;
  }
  return found;
}

class SinkMonotoneBelowMaxPool
    : public impl::SinkMonotoneBelowMaxPoolBase<SinkMonotoneBelowMaxPool> {
public:
  using impl::SinkMonotoneBelowMaxPoolBase<
      SinkMonotoneBelowMaxPool>::SinkMonotoneBelowMaxPoolBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() final {
    SmallVector<linalg::PoolingNhwcMaxOp> pools;
    getOperation().walk(
        [&](linalg::PoolingNhwcMaxOp p) { pools.push_back(p); });
    for (linalg::PoolingNhwcMaxOp p : pools)
      if (p->getBlock())
        (void)sink(p);
  }

private:
  LogicalResult sink(linalg::PoolingNhwcMaxOp pool);
};

LogicalResult SinkMonotoneBelowMaxPool::sink(linalg::PoolingNhwcMaxOp pool) {
  Value padded = pool.getInputs()[0];
  auto padTy = llvm::dyn_cast<MemRefType>(padded.getType());
  auto outTy = llvm::dyn_cast<MemRefType>(pool.getOutputs()[0].getType());
  if (!padTy || !outTy || !padTy.getElementType().isF32() ||
      !outTy.getElementType().isF32() || !padTy.getLayout().isIdentity() ||
      !outTy.getLayout().isIdentity() ||
      !llvm::isa_and_nonnull<memref::AllocOp>(padded.getDefiningOp()))
    return failure();

  // What fills the padding, what copies into it, and nothing else.
  linalg::MapOp fill;
  memref::CopyOp copy;
  memref::SubViewOp window;
  for (Operation *user : padded.getUsers()) {
    if (user == pool.getOperation() || llvm::isa<memref::DeallocOp>(user))
      continue;
    if (auto m = llvm::dyn_cast<linalg::MapOp>(user)) {
      if (fill)
        return failure();
      fill = m;
      continue;
    }
    if (auto sv = llvm::dyn_cast<memref::SubViewOp>(user)) {
      if (window || !sv->hasOneUse())
        return failure();
      window = sv;
      copy = llvm::dyn_cast<memref::CopyOp>(*sv->getUsers().begin());
      if (!copy || copy.getTarget() != sv.getResult())
        return failure();
      continue;
    }
    return failure();
  }
  if (!fill || !copy || !window)
    return failure();
  APFloat padValue(0.0f);
  {
    Operation *y = fill.getBody()->getTerminator();
    if (y->getNumOperands() != 1 || !floatConst(y->getOperand(0), padValue))
      return failure();
  }

  // What the copy reads is the map's output, and the map is monotone.
  Value wide = copy.getSource();
  auto wideTy = llvm::dyn_cast<MemRefType>(wide.getType());
  if (!wideTy || !wideTy.getLayout().isIdentity() ||
      !llvm::isa_and_nonnull<memref::AllocOp>(wide.getDefiningOp()))
    return failure();
  auto map = soleWriter(wide, copy.getOperation());
  if (!map || map.getNumDpsInits() != 1 || map.getOutputs()[0] != wide)
    return failure();
  APFloat floor(0.0f);
  if (!monotoneWithFloor(map, floor))
    return failure();
  // The pad has to be exactly what the map's floor produces, or the padded
  // positions would come back different.
  if (floor.compare(padValue) != APFloat::cmpEqual)
    return failure();

  Value acc = map.getInputs()[0];
  auto accTy = llvm::dyn_cast<MemRefType>(acc.getType());
  if (!accTy || !accTy.getElementType().isInteger(32) ||
      !accTy.getLayout().isIdentity() || accTy.getShape() != wideTy.getShape())
    return failure();
  // Extra operands are per-channel, so the pool carries them through unchanged.
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineMap> maps = map.getIndexingMapsArray();
  unsigned rank = accTy.getRank();
  if (rank != 4 || maps.front().getNumResults() != rank ||
      maps.back().getNumResults() != rank)
    return failure();
  // Not the identity: `--order-loops-for-locality` puts the channel outermost,
  // so both buffers are read through the same permutation -- and a batch of one
  // comes out as a constant zero ([[mlir-unit-axis-is-a-constant-zero]]). What
  // matters is that the accumulator and the output are walked the same way.
  for (unsigned p = 0; p < rank; p++) {
    AffineExpr a = maps.front().getResult(p), o = maps.back().getResult(p);
    if (a == o)
      continue;
    auto c = llvm::dyn_cast<AffineConstantExpr>(a);
    if (!c || c.getValue() != 0 || accTy.getShape()[p] != 1)
      return failure();
  }
  // The channel axis is the last one, and an extra operand may only be read
  // through the iteration dimension that indexes it.
  AffineMap chan =
      AffineMap::get(rank, 0, {maps.back().getResult(rank - 1)}, ctx);
  for (unsigned i = 1; i < map.getNumDpsInputs(); i++)
    if (maps[i] != chan)
      return failure();

  // The pool's output is filled and then written; the fill is dead either way
  // because the padded input is fully initialised.
  linalg::FillOp outFill;
  for (Operation *user : pool.getOutputs()[0].getUsers())
    if (auto f = llvm::dyn_cast<linalg::FillOp>(user)) {
      if (outFill)
        return failure();
      outFill = f;
    }
  if (!outFill)
    return failure();

  // Rewrite, in the accumulator's own type.
  OpBuilder b(fill);
  Location loc = pool.getLoc();
  Type i32 = b.getI32Type();
  Value neutral = b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(i32, APInt::getSignedMinValue(32)));

  Value padI = b.create<memref::AllocOp>(
      loc, MemRefType::get(padTy.getShape(), i32), b.getI64IntegerAttr(64));
  b.create<linalg::FillOp>(loc, ValueRange{neutral}, ValueRange{padI});
  Value winI = b.create<memref::SubViewOp>(
      loc, padI, window.getStaticOffsets(), window.getStaticSizes(),
      window.getStaticStrides());
  b.create<memref::CopyOp>(loc, acc, winI);

  Value pooledI = b.create<memref::AllocOp>(
      loc, MemRefType::get(outTy.getShape(), i32), b.getI64IntegerAttr(64));
  b.create<linalg::FillOp>(loc, ValueRange{neutral}, ValueRange{pooledI});
  Value kernel = b.create<memref::AllocOp>(
      loc,
      MemRefType::get(
          llvm::cast<MemRefType>(pool.getInputs()[1].getType()).getShape(), i32),
      b.getI64IntegerAttr(64));
  b.create<linalg::PoolingNhwcMaxOp>(loc, TypeRange{},
                                     ValueRange{padI, kernel},
                                     ValueRange{pooledI}, pool.getStrides(),
                                     pool.getDilations());

  // The map, now on the pooled accumulator and writing where the pool did.
  SmallVector<Value> ins(map.getInputs().begin(), map.getInputs().end());
  ins[0] = pooledI;
  b.setInsertionPointAfter(pool);
  auto sunk = b.create<linalg::GenericOp>(
      loc, TypeRange{}, ins, ValueRange{pool.getOutputs()[0]}, maps,
      llvm::to_vector(map.getIteratorTypesArray()));
  IRMapping mapping;
  map.getRegion().cloneInto(&sunk.getRegion(), mapping);
  b.setInsertionPointAfter(sunk);
  b.create<memref::DeallocOp>(loc, padI);
  b.create<memref::DeallocOp>(loc, pooledI);
  b.create<memref::DeallocOp>(loc, kernel);

  pool.erase();
  outFill.erase();
  copy.erase();
  window.erase();
  fill.erase();
  map.erase();
  for (Value dead : {padded, wide}) {
    SmallVector<Operation *> users(dead.getUsers().begin(),
                                   dead.getUsers().end());
    for (Operation *u : users)
      if (u->use_empty())
        u->erase();
    if (Operation *def = dead.getDefiningOp())
      if (def->use_empty())
        def->erase();
  }
  return success();
}

} // namespace

} // namespace mlir::gemmlir
