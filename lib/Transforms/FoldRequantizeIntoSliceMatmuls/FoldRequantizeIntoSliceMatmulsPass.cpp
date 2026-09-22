//===- FoldRequantizeIntoSliceMatmulsPass.cpp --------------------*- C++ -*-===//
//
// Many calls into one accumulator, one requantization over the whole of it.
//
// A transformer's attention output is one matmul per head writing its own slice
// of a shared i32 buffer, and a single requantization that reads all of it and
// puts the heads back where the model wants them. The conversion's own
// per-slice fold refuses that, because it needs the destination laid out
// exactly like the accumulator -- so a permutation of whole axes stops it and
// the loop stays on the host, writing its result with a stride.
//
// Each slice has one destination block, and `tiled_matmul_auto` addresses a
// block through a row stride. So every call writes its own i8 and the reader
// goes away entirely.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"

#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_FOLDREQUANTIZEINTOSLICEMATMULS
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

static bool isViewOp(Operation *op) {
  return llvm::isa<memref::ExpandShapeOp, memref::CollapseShapeOp,
                   memref::SubViewOp, memref::CastOp>(op);
}

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

static bool intConst(Value v, int64_t &out) {
  llvm::APInt i;
  if (!matchPattern(v, m_ConstantInt(&i)))
    return false;
  out = i.getSExtValue();
  return true;
}

/// An index the unrolled loop computed rather than wrote down.
///
/// `--unroll-accelerator-loops` leaves each slice's offset as the induction
/// variable's arithmetic -- `addi(lo, muli(step, i))` -- and there is no
/// canonicalization between it and here, so the offsets are `arith` chains and
/// not constants. Folding them here is cheaper than adding a canonicalization
/// to the pipeline, which would rewrite every model's object.
static bool constantIndex(Value v, int64_t &out, int depth = 0) {
  if (depth > 8)
    return false;
  if (intConst(v, out))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  int64_t a = 0, b = 0;
  if (auto add = llvm::dyn_cast<arith::AddIOp>(def)) {
    if (!constantIndex(add.getLhs(), a, depth + 1) ||
        !constantIndex(add.getRhs(), b, depth + 1))
      return false;
    out = a + b;
    return true;
  }
  if (auto mul = llvm::dyn_cast<arith::MulIOp>(def)) {
    if (!constantIndex(mul.getLhs(), a, depth + 1) ||
        !constantIndex(mul.getRhs(), b, depth + 1))
      return false;
    out = a * b;
    return true;
  }
  if (auto sub = llvm::dyn_cast<arith::SubIOp>(def)) {
    if (!constantIndex(sub.getLhs(), a, depth + 1) ||
        !constantIndex(sub.getRhs(), b, depth + 1))
      return false;
    out = a - b;
    return true;
  }
  return false;
}

/// Which iteration dimension each axis is indexed by, or -1 for a unit axis a
/// bufferized map writes as a constant zero
/// ([[mlir-unit-axis-is-a-constant-zero]]). Fails on anything that is not a
/// plain permutation of whole axes.
static bool axisDims(AffineMap map, ArrayRef<int64_t> shape,
                     SmallVectorImpl<int> &out) {
  if ((int64_t)map.getNumResults() != (int64_t)shape.size())
    return false;
  llvm::SmallDenseSet<unsigned> seen;
  out.assign(shape.size(), -1);
  for (unsigned a = 0; a < map.getNumResults(); a++) {
    AffineExpr e = map.getResult(a);
    if (auto d = llvm::dyn_cast<AffineDimExpr>(e)) {
      if (!seen.insert(d.getPosition()).second)
        return false;
      out[a] = (int)d.getPosition();
      continue;
    }
    auto c = llvm::dyn_cast<AffineConstantExpr>(e);
    if (!c || c.getValue() != 0 || shape[a] != 1)
      return false;
  }
  return true;
}

/// The pipeline's own way of folding two constant scales: a division reaches
/// `--combine-constant-scales` already turned into a multiply by the rounded
/// reciprocal, and the two are not the same f32.
static void applyDivisor(APFloat &acc, const APFloat &c) {
  auto rm = APFloat::rmNearestTiesToEven;
  APFloat inv(c.getSemantics(), 1);
  inv.divide(c, rm);
  if (c.isNegative() || !inv.isFiniteNonZero() || inv.isDenormal()) {
    acc.divide(c, rm);
    return;
  }
  acc.multiply(inv, rm);
}

static SmallVector<Operation *> bodyOps(linalg::GenericOp g) {
  SmallVector<Operation *> ops;
  for (Operation &o : g.getBody()->without_terminator())
    ops.push_back(&o);
  return ops;
}

/// The identity, allowing for unit axes a bufferized map writes as a constant
/// zero ([[mlir-unit-axis-is-a-constant-zero]]).
static bool isIdentityOverUnitAxes(AffineMap map, ArrayRef<int64_t> shape) {
  if ((int64_t)map.getNumResults() != (int64_t)shape.size())
    return false;
  for (unsigned i = 0; i < map.getNumResults(); i++) {
    AffineExpr e = map.getResult(i);
    if (e == getAffineDimExpr(i, map.getContext()))
      continue;
    auto c = llvm::dyn_cast<AffineConstantExpr>(e);
    if (c && c.getValue() == 0 && shape[i] == 1)
      continue;
    return false;
  }
  return true;
}

/// `f32 = sitofp(i32) * a`, the dequantization a quantized matmul is followed
/// by. An offset is allowed only when it is zero.
static bool matchDequantize(linalg::GenericOp g, APFloat &a) {
  if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1)
    return false;
  auto inTy = llvm::dyn_cast<MemRefType>(g.getInputs()[0].getType());
  auto outTy = llvm::dyn_cast<MemRefType>(g.getOutputs()[0].getType());
  if (!inTy || !outTy || !inTy.getElementType().isInteger(32) ||
      !outTy.getElementType().isF32() || inTy.getShape() != outTy.getShape())
    return false;
  SmallVector<AffineMap> maps = g.getIndexingMapsArray();
  if (!isIdentityOverUnitAxes(maps[0], inTy.getShape()) ||
      !isIdentityOverUnitAxes(maps.back(), outTy.getShape()))
    return false;

  Value cur = g.getBody()->getArgument(0);
  APFloat scale(1.0f);
  bool sawConvert = false;
  for (Operation *op : bodyOps(g)) {
    APFloat k(0.0f);
    if (auto c = llvm::dyn_cast<arith::SIToFPOp>(op)) {
      if (sawConvert || c.getIn() != cur)
        return false;
      sawConvert = true;
      cur = c.getResult();
    } else if (auto m = llvm::dyn_cast<arith::MulFOp>(op)) {
      if (!sawConvert || m.getLhs() != cur || !floatConst(m.getRhs(), k))
        return false;
      scale.multiply(k, APFloat::rmNearestTiesToEven);
      cur = m.getResult();
    } else if (auto d = llvm::dyn_cast<arith::DivFOp>(op)) {
      if (!sawConvert || d.getLhs() != cur || !floatConst(d.getRhs(), k) ||
          k.isZero())
        return false;
      applyDivisor(scale, k);
      cur = d.getResult();
    } else if (auto s = llvm::dyn_cast<arith::AddFOp>(op)) {
      if (!sawConvert || s.getLhs() != cur || !floatConst(s.getRhs(), k) ||
          !k.isZero())
        return false;
      cur = s.getResult();
    } else {
      return false;
    }
  }
  if (!sawConvert || g.getBody()->getTerminator()->getOperand(0) != cur ||
      !scale.isFiniteNonZero())
    return false;
  a = scale;
  return true;
}

/// `i8 = clamp(roundeven(f32 * c))`. `relu` says the lower bound was zero.
static bool matchRequantize(linalg::GenericOp g, APFloat &c, bool &relu) {
  if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1)
    return false;
  auto inTy = llvm::dyn_cast<MemRefType>(g.getInputs()[0].getType());
  auto outTy = llvm::dyn_cast<MemRefType>(g.getOutputs()[0].getType());
  if (!inTy || !outTy || !inTy.getElementType().isF32() ||
      !outTy.getElementType().isInteger(8))
    return false;

  Value cur = g.getBody()->getArgument(0);
  APFloat scale(1.0f);
  bool sawRound = false, sawCast = false, sawTrunc = false;
  bool sawLow = false, sawHigh = false;
  relu = false;
  for (Operation *op : bodyOps(g)) {
    APFloat k(0.0f);
    int64_t n = 0;
    if (auto m = llvm::dyn_cast<arith::MulFOp>(op)) {
      if (sawRound || m.getLhs() != cur || !floatConst(m.getRhs(), k))
        return false;
      scale.multiply(k, APFloat::rmNearestTiesToEven);
      cur = m.getResult();
    } else if (auto d = llvm::dyn_cast<arith::DivFOp>(op)) {
      if (sawRound || d.getLhs() != cur || !floatConst(d.getRhs(), k) ||
          k.isZero())
        return false;
      applyDivisor(scale, k);
      cur = d.getResult();
    } else if (auto r = llvm::dyn_cast<math::RoundEvenOp>(op)) {
      if (sawRound || r.getOperand() != cur)
        return false;
      sawRound = true;
      cur = r.getResult();
    } else if (auto f = llvm::dyn_cast<arith::FPToSIOp>(op)) {
      if (!sawRound || sawCast || f.getIn() != cur)
        return false;
      sawCast = true;
      cur = f.getResult();
    } else if (auto mx = llvm::dyn_cast<arith::MaxSIOp>(op)) {
      if (!sawCast || sawLow || mx.getLhs() != cur || !intConst(mx.getRhs(), n))
        return false;
      if (n == 0)
        relu = true;
      else if (n != -128)
        return false;
      sawLow = true;
      cur = mx.getResult();
    } else if (auto mn = llvm::dyn_cast<arith::MinSIOp>(op)) {
      if (!sawCast || sawHigh || mn.getLhs() != cur ||
          !intConst(mn.getRhs(), n) || n != 127)
        return false;
      sawHigh = true;
      cur = mn.getResult();
    } else if (auto t = llvm::dyn_cast<arith::TruncIOp>(op)) {
      if (!sawLow || !sawHigh || sawTrunc || t.getIn() != cur)
        return false;
      sawTrunc = true;
      cur = t.getResult();
    } else {
      return false;
    }
  }
  if (!sawTrunc || g.getBody()->getTerminator()->getOperand(0) != cur ||
      !scale.isFiniteNonZero())
    return false;
  c = scale;
  return true;
}

/// One writer: the call, and which index along the accumulator's first axis it
/// writes.
struct Writer {
  MatMulInt8Op call;
  memref::SubViewOp slice;
  int64_t index;
};

class FoldRequantizeIntoSliceMatmuls
    : public impl::FoldRequantizeIntoSliceMatmulsBase<
          FoldRequantizeIntoSliceMatmuls> {
public:
  using impl::FoldRequantizeIntoSliceMatmulsBase<
      FoldRequantizeIntoSliceMatmuls>::FoldRequantizeIntoSliceMatmulsBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect, GemmlirDialect>();
  }

  void runOnOperation() final {
    SmallVector<memref::AllocOp> buffers;
    getOperation().walk([&](memref::AllocOp a) {
      auto ty = llvm::dyn_cast<MemRefType>(a.getType());
      if (ty && ty.getRank() == 3 && ty.getElementType().isInteger(32) &&
          ty.hasStaticShape())
        buffers.push_back(a);
    });
    for (memref::AllocOp a : buffers)
      if (a->getBlock())
        (void)fold(a);
  }

private:
  LogicalResult fold(memref::AllocOp accOp);
};

LogicalResult FoldRequantizeIntoSliceMatmuls::fold(memref::AllocOp accOp) {
  Value acc = accOp.getResult();
  auto accTy = llvm::cast<MemRefType>(acc.getType());
  const int64_t H = accTy.getShape()[0], M = accTy.getShape()[1],
                N = accTy.getShape()[2];

  // The writers: one call per slice, at a constant index, and nothing else.
  SmallVector<Writer> writers;
  linalg::GenericOp deq;
  for (Operation *user : acc.getUsers()) {
    if (llvm::isa<memref::DeallocOp>(user))
      continue;
    if (auto sv = llvm::dyn_cast<memref::SubViewOp>(user)) {
      if (!sv->hasOneUse())
        return failure();
      auto call = llvm::dyn_cast<MatMulInt8Op>(*sv->getUsers().begin());
      if (!call || call.getOutMat() != sv.getResult() || call.getAccumulate() ||
          call.getBias())
        return failure();
      SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
      SmallVector<OpFoldResult> sizes = sv.getMixedSizes();
      if (offs.size() != 3 || sizes.size() != 3)
        return failure();
      int64_t index = 0;
      if (auto attr = llvm::dyn_cast<Attribute>(offs[0])) {
        index = llvm::cast<IntegerAttr>(attr).getInt();
      } else if (!constantIndex(llvm::cast<Value>(offs[0]), index)) {
        return failure();
      }
      auto isConst = [&](OpFoldResult r, int64_t want) {
        if (auto attr = llvm::dyn_cast<Attribute>(r))
          return llvm::cast<IntegerAttr>(attr).getInt() == want;
        int64_t v = 0;
        return constantIndex(llvm::cast<Value>(r), v) && v == want;
      };
      if (!isConst(offs[1], 0) || !isConst(offs[2], 0) ||
          !isConst(sizes[0], 1) || !isConst(sizes[1], M) ||
          !isConst(sizes[2], N))
        return failure();
      for (OpFoldResult s : sv.getMixedStrides())
        if (!isConst(s, 1))
          return failure();
      if (index < 0 || index >= H)
        return failure();
      writers.push_back(Writer{call, sv, index});
      continue;
    }
    if (auto g = llvm::dyn_cast<linalg::GenericOp>(user)) {
      if (deq || g.getNumDpsInputs() != 1 || g.getInputs()[0] != acc)
        return failure();
      deq = g;
      continue;
    }
    return failure();
  }
  if (writers.size() < 2 || !deq)
    return failure();
  {
    llvm::SmallDenseSet<int64_t> seen;
    for (Writer &w : writers)
      if (!seen.insert(w.index).second)
        return failure();
    if ((int64_t)seen.size() != H)
      return failure();
  }

  APFloat dequantScale(1.0f);
  if (!matchDequantize(deq, dequantScale))
    return failure();

  // The f32 buffer is read once, by a requantization that may also move whole
  // axes around.
  Value mid = deq.getOutputs()[0];
  if (!llvm::isa_and_nonnull<memref::AllocOp>(mid.getDefiningOp()))
    return failure();
  linalg::GenericOp req;
  Value readAs;
  {
    SmallVector<Value> work{mid};
    while (!work.empty()) {
      Value v = work.pop_back_val();
      for (Operation *user : v.getUsers()) {
        if (user == deq.getOperation() || llvm::isa<memref::DeallocOp>(user))
          continue;
        if (isViewOp(user)) {
          work.push_back(user->getResult(0));
          continue;
        }
        auto g = llvm::dyn_cast<linalg::GenericOp>(user);
        if (req || !g || g.getNumDpsInputs() != 1 || g.getInputs()[0] != v)
          return failure();
        req = g;
        readAs = v;
      }
    }
  }
  if (!req)
    return failure();
  APFloat requantScale(1.0f);
  bool relu = false;
  if (!matchRequantize(req, requantScale, relu))
    return failure();
  APFloat total = dequantScale;
  total.multiply(requantScale, APFloat::rmNearestTiesToEven);
  if (!total.isFiniteNonZero())
    return failure();

  // The reader's view of the f32 buffer has the same axes in the same order,
  // with unit axes allowed; the map over it has to be the identity so that
  // iteration dimension and axis agree.
  auto readTy = llvm::cast<MemRefType>(readAs.getType());
  Value out = req.getOutputs()[0];
  auto outTy = llvm::cast<MemRefType>(out.getType());
  SmallVector<AffineMap> maps = req.getIndexingMapsArray();
  SmallVector<int> inDim, outDim;
  if (!axisDims(maps[0], readTy.getShape(), inDim) ||
      !axisDims(maps.back(), outTy.getShape(), outDim))
    return failure();

  // The reader's view of the f32 buffer has the accumulator's three axes in
  // the same order, with unit axes allowed anywhere.
  SmallVector<int64_t> nonUnit;
  for (unsigned a = 0; a < readTy.getRank(); a++)
    if (readTy.getShape()[a] != 1)
      nonUnit.push_back(a);
  if (nonUnit.size() != 3 || readTy.getShape()[nonUnit[0]] != H ||
      readTy.getShape()[nonUnit[1]] != M || readTy.getShape()[nonUnit[2]] != N)
    return failure();
  int sliceDim = inDim[nonUnit[0]];
  if (sliceDim < 0)
    return failure();

  // Where the slice axis went in the output. Either side may be the one that
  // carries the permutation.
  int64_t sliceAxis = -1;
  for (unsigned a = 0; a < outTy.getRank(); a++)
    if (outDim[a] == sliceDim) {
      sliceAxis = a;
      break;
    }
  if (sliceAxis < 0 || outTy.getShape()[sliceAxis] != H)
    return failure();

  // The output's allocation has to exist before the first call that will write
  // it. Bufferization puts it after the loop, next to the requantization; an
  // allocation reads nothing, so moving it up is free. Settle this before
  // anything is created, so a refusal leaves the function as it was.
  Operation *first = writers.front().call;
  for (Writer &w : writers)
    if (w.call->isBeforeInBlock(first))
      first = w.call;
  if (Operation *def = out.getDefiningOp()) {
    if (def->getBlock() != first->getBlock())
      return failure();
    if (!def->isBeforeInBlock(first)) {
      if (!llvm::isa<memref::AllocOp>(def) || !def->getOperands().empty())
        return failure();
      def->moveBefore(first);
    }
  }

  // Rewrite: each call writes its own block of the output, requantized on the
  // way out.
  SmallVector<memref::SubViewOp> newSlices;
  {
    // One slice decides the shape for all of them, so settle it before the
    // first rewrite rather than half way through. The block the call writes
    // has to survive rank reduction as exactly `M x N` in that order --
    // a permutation that also swapped the matmul's own two axes would give
    // `N x M`, which is not the same memory and not addressable as a stride.
    SmallVector<int64_t> kept;
    for (unsigned a = 0; a < outTy.getRank(); a++)
      if ((int64_t)a != sliceAxis && outTy.getShape()[a] != 1)
        kept.push_back(outTy.getShape()[a]);
    if (kept.size() != 2 || kept[0] != M || kept[1] != N)
      return failure();

    OpBuilder b(first);
    SmallVector<OpFoldResult> offs, sizes, strides;
    for (unsigned a = 0; a < outTy.getRank(); a++) {
      offs.push_back(b.getIndexAttr(0));
      sizes.push_back(
          b.getIndexAttr((int64_t)a == sliceAxis ? 1 : outTy.getShape()[a]));
      strides.push_back(b.getIndexAttr(1));
    }
    auto sliceTy = llvm::dyn_cast<MemRefType>(
        memref::SubViewOp::inferRankReducedResultType({M, N}, outTy, offs,
                                                      sizes, strides));
    SmallVector<int64_t> sliceStrides;
    int64_t sliceOffset;
    if (!sliceTy || sliceTy.getShape() != ArrayRef<int64_t>({M, N}) ||
        failed(sliceTy.getStridesAndOffset(sliceStrides, sliceOffset)) ||
        sliceStrides.size() != 2 || sliceStrides[1] != 1 ||
        ShapedType::isDynamic(sliceStrides[0]))
      return failure();
  }
  for (Writer &w : writers) {
    OpBuilder b(w.call);
    SmallVector<OpFoldResult> offs, sizes, strides;
    for (unsigned a = 0; a < outTy.getRank(); a++) {
      offs.push_back(b.getIndexAttr((int64_t)a == sliceAxis ? w.index : 0));
      sizes.push_back(b.getIndexAttr((int64_t)a == sliceAxis
                                         ? 1
                                         : outTy.getShape()[a]));
      strides.push_back(b.getIndexAttr(1));
    }
    auto sliceTy = llvm::dyn_cast<MemRefType>(
        memref::SubViewOp::inferRankReducedResultType({M, N}, outTy, offs, sizes,
                                                      strides));
    if (!sliceTy)
      return failure();
    SmallVector<int64_t> sliceStrides;
    int64_t sliceOffset;
    if (failed(sliceTy.getStridesAndOffset(sliceStrides, sliceOffset)) ||
        sliceStrides.size() != 2 || sliceStrides[1] != 1 ||
        ShapedType::isDynamic(sliceStrides[0]))
      return failure();
    Value dst = b.create<memref::SubViewOp>(w.call.getLoc(), sliceTy, out, offs,
                                            sizes, strides);
    newSlices.push_back(llvm::cast<memref::SubViewOp>(dst.getDefiningOp()));
    b.create<MatMulInt8ScaleOp>(
        w.call.getLoc(), w.call.getLhsMat(), w.call.getRhsMat(), dst,
        /*bias=*/Value(), w.call.getLhsScaleAttr(), w.call.getRhsScaleAttr(),
        w.call.getTransposeLhsAttr(), w.call.getTransposeRhsAttr(),
        b.getF32FloatAttr(total.convertToFloat()), b.getF32FloatAttr(1.0f),
        ActAttr::get(b.getContext(), relu ? Act::RELU : Act::NONE),
        w.call.getDataflowAttr());
  }

  req.erase();
  deq.erase();
  for (Writer &w : writers) {
    w.call.erase();
    w.slice.erase();
  }
  SmallVector<Operation *> dead;
  for (Value buf : {mid, acc})
    for (Operation *user : buf.getUsers())
      dead.push_back(user);
  for (Operation *op : llvm::reverse(dead))
    if (op->use_empty())
      op->erase();
  for (Value buf : {mid, acc}) {
    SmallVector<Operation *> more;
    for (Operation *user : buf.getUsers())
      more.push_back(user);
    for (Operation *op : more)
      if (op->use_empty())
        op->erase();
    if (Operation *def = buf.getDefiningOp())
      if (def->use_empty())
        def->erase();
  }
  return success();
}

} // namespace

} // namespace mlir::gemmlir
