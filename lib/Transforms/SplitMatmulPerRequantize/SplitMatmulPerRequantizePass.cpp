//===- SplitMatmulPerRequantizePass.cpp --------------------------*- C++ -*-===//
//
// One accumulator, three scales: give each its own call.
//
// A quantized matmul whose i32 accumulator one requantization reads becomes a
// `matmul_i8_scale` and the loop goes away. A transformer's attention
// projection never gets there: one matmul writes Q, K and V side by side and
// three readers take a third of the columns each, at three different scales.
// One buffer, three scales, so no single call can serve it.
//
// A column slice of the product is the same rows against a column slice of the
// weights, and `tiled_matmul_auto` reads its operands through a row stride --
// so the slice costs a `memref.subview` and no copy at all.
//
// On `vit_tiny`: twelve matmuls become thirty-six calls, the host's
// element-visits go 363,088 -> **284,752** (-21.6%) and the heaviest single
// shape, 43.1% of what was left, disappears. What the readers still do is the
// relayout into heads they were doing anyway, now a byte copy -- and two thirds
// of those the following passes absorb into their own readers. The flush count
// does not move (74 before and after): `--place-cache-flushes` runs after this
// and proves all seventy-two new ones unnecessary.
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

#define GEN_PASS_DEF_SPLITMATMULPERREQUANTIZE
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

/// Multiply by a constant the way the rest of the pipeline will.
///
/// `--reciprocal-for-division` runs before `--combine-constant-scales`, so a
/// division by a constant reaches the fold already turned into a multiply by
/// the rounded reciprocal -- and the two are not the same f32. This has to be
/// the number the host would have multiplied by, or the accelerator rounds a
/// different way on the elements that sit on a tie.
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

/// `f32 = sitofp(i32) * a`. An offset is allowed only when it is zero: the
/// accelerator's output pipeline has a scale and no offset.
static bool matchDequantize(linalg::GenericOp g, APFloat &a) {
  if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1)
    return false;
  for (AffineMap m : g.getIndexingMapsArray())
    if (!m.isIdentity())
      return false;
  auto inTy = llvm::dyn_cast<MemRefType>(g.getInputs()[0].getType());
  auto outTy = llvm::dyn_cast<MemRefType>(g.getOutputs()[0].getType());
  if (!inTy || !outTy || !inTy.getElementType().isInteger(32) ||
      !outTy.getElementType().isF32())
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

/// `i8 = clamp(roundeven(f32 * c))`. `relu` says the lower bound was zero,
/// which is what the accelerator's RELU activation gives.
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

/// A window on an `M x N` buffer that is every row and one run of columns.
///
/// The view's own strides say it: the leading dimensions walk whole rows -- a
/// stride that is a multiple of N, a shape that multiplies out to M -- and the
/// trailing ones a contiguous run inside a row. `split` is where the two parts
/// meet, which is also where the replacement's reassociation goes.
struct ColumnWindow {
  int64_t offset, width;
  unsigned split;
};

static std::optional<ColumnWindow> columnWindow(MemRefType ty, int64_t M,
                                                int64_t N) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(ty.getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset))
    return std::nullopt;
  for (int64_t s : strides)
    if (ShapedType::isDynamic(s))
      return std::nullopt;
  ArrayRef<int64_t> shape = ty.getShape();

  for (unsigned p = 1; p < shape.size(); p++) {
    int64_t rows = 1, cols = 1;
    for (unsigned d = 0; d < p; d++)
      rows *= shape[d];
    for (unsigned d = p; d < shape.size(); d++)
      cols *= shape[d];
    if (rows != M || cols > N)
      continue;
    bool ok = true;
    int64_t run = N;
    for (unsigned d = p; d-- > 0;) {
      if (shape[d] != 1 && strides[d] != run)
        ok = false;
      run *= shape[d];
    }
    run = 1;
    for (unsigned d = shape.size(); d-- > p;) {
      if (shape[d] != 1 && strides[d] != run)
        ok = false;
      run *= shape[d];
    }
    if (!ok || offset < 0 || offset + cols > N)
      continue;
    return ColumnWindow{offset, cols, p};
  }
  return std::nullopt;
}

struct Reader {
  linalg::GenericOp generic;
  Value view;
  ColumnWindow window;
  APFloat scale;
  bool relu;
};

/// The generics that read `buf`, through any chain of views. Fails when
/// anything else does.
static bool readersOf(Value buf, Operation *skip,
                      SmallVectorImpl<std::pair<linalg::GenericOp, Value>> &out,
                      SmallVectorImpl<Operation *> &views) {
  SmallVector<Value> work{buf};
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (user == skip || llvm::isa<memref::DeallocOp>(user))
        continue;
      if (isViewOp(user)) {
        views.push_back(user);
        work.push_back(user->getResult(0));
        continue;
      }
      auto g = llvm::dyn_cast<linalg::GenericOp>(user);
      if (!g || g.getNumDpsInputs() != 1 || g.getInputs()[0] != v)
        return false;
      // Reading it is what makes it a reader; one that also writes it is not
      // the shape this is about.
      if (llvm::is_contained(g.getOutputs(), v))
        return false;
      out.push_back({g, v});
    }
  }
  return true;
}

class SplitMatmulPerRequantize
    : public impl::SplitMatmulPerRequantizeBase<SplitMatmulPerRequantize> {
public:
  using impl::SplitMatmulPerRequantizeBase<
      SplitMatmulPerRequantize>::SplitMatmulPerRequantizeBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect, GemmlirDialect>();
  }

  void runOnOperation() final {
    SmallVector<MatMulInt8Op> candidates;
    getOperation().walk([&](MatMulInt8Op op) { candidates.push_back(op); });
    for (MatMulInt8Op mm : candidates)
      (void)split(mm);
  }

private:
  LogicalResult split(MatMulInt8Op mm);
};

LogicalResult SplitMatmulPerRequantize::split(MatMulInt8Op mm) {
  if (mm.getAccumulate() || mm.getTransposeRhs())
    return failure();
  Value acc = mm.getOutMat();
  if (!llvm::isa_and_nonnull<memref::AllocOp>(acc.getDefiningOp()))
    return failure();
  auto accTy = llvm::cast<MemRefType>(acc.getType());
  if (!accTy.getLayout().isIdentity())
    return failure();
  const int64_t M = accTy.getShape()[0], N = accTy.getShape()[1];

  SmallVector<Operation *> stale;
  SmallVector<std::pair<linalg::GenericOp, Value>> accReaders;
  if (!readersOf(acc, mm, accReaders, stale) || accReaders.size() != 1 ||
      accReaders[0].second != acc)
    return failure();
  linalg::GenericOp deq = accReaders[0].first;
  APFloat dequantScale(1.0f);
  if (!matchDequantize(deq, dequantScale))
    return failure();

  Value mid = deq.getOutputs()[0];
  if (!llvm::isa_and_nonnull<memref::AllocOp>(mid.getDefiningOp()))
    return failure();

  SmallVector<std::pair<linalg::GenericOp, Value>> midReaders;
  if (!readersOf(mid, deq, midReaders, stale) || midReaders.size() < 2)
    return failure();

  SmallVector<Reader> readers;
  for (auto [g, v] : midReaders) {
    std::optional<ColumnWindow> win =
        columnWindow(llvm::cast<MemRefType>(v.getType()), M, N);
    APFloat c(1.0f);
    bool relu = false;
    if (!win || !matchRequantize(g, c, relu))
      return failure();
    APFloat total = dequantScale;
    total.multiply(c, APFloat::rmNearestTiesToEven);
    if (!total.isFiniteNonZero())
      return failure();
    readers.push_back(Reader{g, v, *win, total, relu});
  }

  // No two readers on the same columns, and together no wider than the
  // accumulator: the split must never do the arithmetic twice.
  int64_t covered = 0;
  for (unsigned i = 0; i < readers.size(); i++) {
    covered += readers[i].window.width;
    for (unsigned j = i + 1; j < readers.size(); j++) {
      const ColumnWindow &a = readers[i].window, &b = readers[j].window;
      if (a.offset < b.offset + b.width && b.offset < a.offset + a.width)
        return failure();
    }
  }
  if (covered > N)
    return failure();

  OpBuilder b(mm);
  Location loc = mm.getLoc();
  Type i8 = b.getI8Type();
  Value rhs = mm.getRhsMat();
  const int64_t K = llvm::cast<MemRefType>(rhs.getType()).getShape()[0];
  Value bias = mm.getBias();

  SmallVector<Value> temps;
  for (Reader &r : readers) {
    Value rhsSlice = rhs;
    if (r.window.offset != 0 || r.window.width != N)
      rhsSlice = b.create<memref::SubViewOp>(
          loc, rhs, ArrayRef<int64_t>{0, r.window.offset},
          ArrayRef<int64_t>{K, r.window.width}, ArrayRef<int64_t>{1, 1});
    Value biasSlice = bias;
    if (bias && (r.window.offset != 0 || r.window.width != N)) {
      auto biasTy = llvm::cast<MemRefType>(bias.getType());
      biasSlice = b.create<memref::SubViewOp>(
          loc, bias, ArrayRef<int64_t>{0, r.window.offset},
          ArrayRef<int64_t>{biasTy.getShape()[0], r.window.width},
          ArrayRef<int64_t>{1, 1});
    }
    Value tmp = b.create<memref::AllocOp>(
        loc, MemRefType::get({M, r.window.width}, i8));
    temps.push_back(tmp);
    b.create<MatMulInt8ScaleOp>(
        loc, mm.getLhsMat(), rhsSlice, tmp, biasSlice, mm.getLhsScaleAttr(),
        mm.getRhsScaleAttr(), mm.getTransposeLhsAttr(),
        mm.getTransposeRhsAttr(), b.getF32FloatAttr(r.scale.convertToFloat()),
        b.getF32FloatAttr(1.0f),
        ActAttr::get(b.getContext(), r.relu ? Act::RELU : Act::NONE),
        mm.getDataflowAttr());
  }

  // What the reader still has to do is the relayout it was doing anyway, now
  // over bytes: same maps, same output, a body that only yields.
  for (auto [r, tmp] : llvm::zip(readers, temps)) {
    OpBuilder rb(r.generic);
    auto viewTy = llvm::cast<MemRefType>(r.view.getType());
    ArrayRef<int64_t> shape = viewTy.getShape();
    Value src = tmp;
    if (!(shape.size() == 2 && shape[0] == M && shape[1] == r.window.width)) {
      ReassociationIndices rows, cols;
      for (unsigned d = 0; d < r.window.split; d++)
        rows.push_back(d);
      for (unsigned d = r.window.split; d < shape.size(); d++)
        cols.push_back(d);
      src = rb.create<memref::ExpandShapeOp>(
          r.generic.getLoc(), MemRefType::get(shape, i8), tmp,
          SmallVector<ReassociationIndices>{rows, cols});
    }
    rb.create<linalg::GenericOp>(
        r.generic.getLoc(), TypeRange{}, ValueRange{src},
        ValueRange{r.generic.getOutputs()[0]}, r.generic.getIndexingMapsArray(),
        llvm::to_vector(r.generic.getIteratorTypesArray()),
        [](OpBuilder &nb, Location nl, ValueRange args) {
          nb.create<linalg::YieldOp>(nl, args[0]);
        });
    r.generic.erase();
  }

  // The temporaries die where the buffer they replaced did.
  for (Operation *user : mid.getUsers())
    if (llvm::isa<memref::DeallocOp>(user)) {
      OpBuilder fb(user);
      for (Value tmp : temps)
        fb.create<memref::DeallocOp>(user->getLoc(), tmp);
      break;
    }

  deq.erase();
  mm.erase();

  // Take the two buffers away here rather than leaving them to a
  // canonicalization: adding one to the pipeline for this would rewrite every
  // model's object, including the twelve this pass never fires on.
  for (Operation *view : llvm::reverse(stale))
    if (view->use_empty())
      view->erase();
  for (Value buf : {mid, acc}) {
    SmallVector<Operation *> frees;
    for (Operation *user : buf.getUsers())
      frees.push_back(user);
    for (Operation *free : frees)
      if (llvm::isa<memref::DeallocOp>(free))
        free->erase();
    if (Operation *alloc = buf.getDefiningOp())
      if (alloc->use_empty())
        alloc->erase();
  }
  return success();
}

} // namespace

} // namespace mlir::gemmlir
