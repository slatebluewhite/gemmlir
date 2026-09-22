//===- LinalgToGemmlirPass.cpp - Linalg to Gemmlir --------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "Gemmlir/GemmlirDialect.h"
#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirPasses.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_CONVERTLINALGTOGEMMLIR
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {
/// See `rowsFollowTheStride` in lib/Gemmlir/GemmlirOps.cpp: the runtime derives
/// a row from the single stride it takes between two pixels, so a window
/// narrower than the buffer around it cannot be expressed. Checked here as well
/// as in the verifier, so such a convolution is left as a loop instead of
/// building an operation that will not verify.
static bool convBufferIsWalkable(Value v) {
  auto t = llvm::dyn_cast<MemRefType>(v.getType());
  if (!t || t.getRank() != 4)
    return false;
  int64_t offset = 0;
  SmallVector<int64_t> strides;
  if (failed(t.getStridesAndOffset(strides, offset)) || strides.size() != 4 ||
      strides[3] != 1)
    return false;
  return strides[1] == t.getShape()[2] * strides[2] &&
         strides[0] == t.getShape()[1] * strides[1];
}


/// The `linalg.fill` that proves `buf` holds zeros where `user` runs, if any.
///
/// `linalg.matmul` accumulates into its output operand, so the lowering has to
/// feed the buffer back to the accelerator as the bias operand. That costs a
/// read of the whole output tile, which is pure waste when the buffer was just
/// zero-filled -- the shape the `--quantize` pipeline produces. Only the
/// immediately preceding `linalg.fill` of a zero constant counts; anything else
/// that touches the buffer in between gives up, since a wrong answer here is a
/// silently dropped accumulator.
static linalg::FillOp zeroFillFor(Value buf, Operation *user) {
  for (Operation *prev = user->getPrevNode(); prev; prev = prev->getPrevNode()) {
    if (auto fill = llvm::dyn_cast<linalg::FillOp>(prev)) {
      if (fill.getOutputs().size() == 1 && fill.getOutputs()[0] == buf &&
          fill.getInputs().size() == 1 &&
          matchPattern(fill.getInputs()[0], m_Zero()))
        return fill;
    }
    if (llvm::is_contained(prev->getOperands(), buf))
      return {};
  }
  return {};
}

/// Drop the fill `zeroFillFor` found, once the matmul replacing `user` is in
/// place and does not accumulate.
///
/// Such a matmul lowers with `D = NULL` and `full_C`, so `tiled_matmul_auto`
/// *writes* every element of the output rather than adding to it: the zeros are
/// never read. Leaving the fill in costs a store per output element on every
/// inference -- 2874 of them on the two-layer CNN, for four matmuls. Only the
/// bufferized form is erased; a fill that still yields a tensor has a result
/// other operations may be using.
static void eraseDeadZeroFill(PatternRewriter &rewriter, linalg::FillOp fill) {
  if (fill && fill->getNumResults() == 0)
    rewriter.eraseOp(fill);
}

/// True when `m` reads element `i` of the iteration space at iteration `i`.
///
/// That is the identity, except that a frontend writes a constant 0 rather than
/// the dimension wherever an axis has extent 1 -- `(d0, d1) -> (0, d1)` for a
/// single-row matmul, which reads exactly what the identity would.
static bool isWholeShapeRead(AffineMap m, ArrayRef<int64_t> shape) {
  if (m.getNumResults() != shape.size() || m.getNumDims() != shape.size())
    return false;
  for (auto [r, e] : llvm::enumerate(m.getResults())) {
    if (auto dim = llvm::dyn_cast<AffineDimExpr>(e)) {
      if (dim.getPosition() != r)
        return false;
      continue;
    }
    auto cst = llvm::dyn_cast<AffineConstantExpr>(e);
    if (!cst || cst.getValue() != 0 || shape[r] != 1)
      return false;
  }
  return true;
}

/// The next operation after `op` that touches `buf`.
///
/// Bufferization puts the destination's `memref.alloc` between the matmul and
/// the operation that consumes it, so "the next operation" is not the one that
/// matters. Anything that does not name the buffer cannot have changed it.
/// The allocation a chain of reshapes names. A `memref.collapse_shape` or
/// `memref.expand_shape` is the same bytes in the same order under another
/// shape, so the two views are interchangeable as memory -- unlike a subview,
/// which is only part of it, and so is not walked through here.
static SmallVector<ReassociationIndices> asMatrix(ArrayRef<int64_t> shape);

static Value reshapeBaseOf(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto op = llvm::dyn_cast<memref::CollapseShapeOp>(def)) {
      v = op.getSrc();
      continue;
    }
    if (auto op = llvm::dyn_cast<memref::ExpandShapeOp>(def)) {
      v = op.getSrc();
      continue;
    }
    break;
  }
  return v;
}

/// The next operation that touches the memory `base` names, by whichever of
/// its reshapes. Building a view is not touching it.
static Operation *nextUseOfBuffer(Operation *op, Value base) {
  for (Operation *n = op->getNextNode(); n; n = n->getNextNode()) {
    if (llvm::isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(n))
      continue;
    for (Value v : n->getOperands())
      if (reshapeBaseOf(v) == base)
        return n;
  }
  return nullptr;
}

static Operation *nextUseOf(Operation *op, Value buf) {
  for (Operation *n = op->getNextNode(); n; n = n->getNextNode())
    if (llvm::is_contained(n->getOperands(), buf))
      return n;
  return nullptr;
}

/// True when nothing strictly between `from` and `to` names any of `values`.
/// The memrefs an operation writes, or nothing when that is not known.
///
/// Every accelerator op writes exactly one operand -- the one it names
/// `output` (`outMat` on the matmuls) -- and only reads the rest.
static std::optional<SmallVector<Value>> writtenOperandsOf(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<SmallVector<Value>>>(op)
      .Case<MatMulInt8Op>([](auto o) {
        return SmallVector<Value>{o.getOutMat()};
      })
      .Case<MatMulInt8ScaleOp>([](auto o) {
        return SmallVector<Value>{o.getOutMat()};
      })
      .Case<ResAddInt8Op>([](auto o) {
        return SmallVector<Value>{o.getOutMat()};
      })
      .Case<NormInt8Op>([](auto o) {
        return SmallVector<Value>{o.getOutput()};
      })
      .Case<Conv2DInt8Op, DepthwiseConv2DInt8Op>([](auto o) {
        return SmallVector<Value>{o.getOutput()};
      })
      .Case<memref::CopyOp>([](auto o) {
        return SmallVector<Value>{o.getTarget()};
      })
      .Case<DestinationStyleOpInterface>([](auto o) {
        return SmallVector<Value>(o.getDpsInits());
      })
      .Default([](Operation *) { return std::nullopt; });
}

/// Whether `values` still hold at `to` what they held at `from`.
///
/// Reading one in between is not a reason to refuse: a fold that moves an
/// operation down to `to` does the same reads it always did. Only a write
/// between the two changes what it would see. Three projections of a
/// transformer share one activation and are emitted back to back, so each of
/// them reads the operands of the others -- refusing on a read alone left every
/// one of their requantizations unfolded.
static bool untouchedBetween(Operation *from, Operation *to,
                             ArrayRef<Value> values) {
  for (Operation *n = from->getNextNode(); n && n != to; n = n->getNextNode()) {
    std::optional<SmallVector<Value>> written = writtenOperandsOf(n);
    for (Value v : values) {
      if (!llvm::is_contained(n->getOperands(), v))
        continue;
      if (!written || llvm::is_contained(*written, v))
        return false;
    }
  }
  return true;
}

/// Reads a `linalg.matmul`'s access maps as a pair of transpose flags.
///
/// MLIR 22 expresses a transposed matmul by overriding `indexing_maps` rather
/// than with a separate op: A is `(m,n,k)->(m,k)` normally and `(m,n,k)->(k,m)`
/// transposed, B is `(k,n)` / `(n,k)`. Ignoring the attribute would quietly
/// compute the untransposed product, so anything not recognised -- a broadcast
/// map, a permuted result -- fails the match instead.
static std::optional<std::pair<bool, bool>>
readTransposes(linalg::MatmulOp matmul) {
  SmallVector<AffineMap> maps = matmul.getIndexingMapsArray();
  if (maps.size() != 3)
    return std::nullopt;

  MLIRContext *ctx = matmul.getContext();
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  auto map = [&](AffineExpr a, AffineExpr b) {
    return AffineMap::get(3, 0, {a, b}, ctx);
  };

  bool transposeLhs;
  if (maps[0] == map(m, k))
    transposeLhs = false;
  else if (maps[0] == map(k, m))
    transposeLhs = true;
  else
    return std::nullopt;

  bool transposeRhs;
  if (maps[1] == map(k, n))
    transposeRhs = false;
  else if (maps[1] == map(n, k))
    transposeRhs = true;
  else
    return std::nullopt;

  if (maps[2] != map(m, n))
    return std::nullopt;
  return std::make_pair(transposeLhs, transposeRhs);
}

/// Matches a bufferized relu over `buf`: an elementwise `linalg.generic` whose
/// body is `arith.maxsi(%in, 0)`, reading and writing the same buffer.
static bool isReluInPlace(Operation *op, Value buf) {
  auto generic = llvm::dyn_cast<linalg::GenericOp>(op);
  if (!generic || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1)
    return false;
  if (generic.getInputs()[0] != buf || generic.getOutputs()[0] != buf)
    return false;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;
  if (!llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap m) { return m.isIdentity(); }))
    return false;

  Block &body = generic.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto max = yield.getOperand(0).getDefiningOp<arith::MaxSIOp>();
  if (!max)
    return false;
  Value in = body.getArgument(0);
  Value other;
  if (max.getLhs() == in)
    other = max.getRhs();
  else if (max.getRhs() == in)
    other = max.getLhs();
  else
    return false;
  return matchPattern(other, m_Zero());
}

// linalg.matmul -> gemmlir.matmul_i8 / gemmlir.matmul_i8_scale.
class LinalgMatmulOpToGemmlir : public OpConversionPattern<linalg::MatmulOp> {
public:
  LinalgMatmulOpToGemmlir(MLIRContext *ctx, Dataflow dataflow)
      : OpConversionPattern(ctx), dataflow(dataflow) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const final {
    ValueRange inputs = matmul.getInputs();
    ValueRange outputs = matmul.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return rewriter.notifyMatchFailure(matmul, "expected two inputs and one output");

    for (Value v : inputs) {
      auto mt = llvm::dyn_cast<MemRefType>(v.getType());
      if (!mt || !mt.getElementType().isInteger(8))
        return rewriter.notifyMatchFailure(matmul, "inputs must be memref<...xi8>");
    }

    Value lhs = inputs[0], rhs = inputs[1], out = outputs[0];
    auto outType = llvm::dyn_cast<MemRefType>(out.getType());
    if (!outType)
      return rewriter.notifyMatchFailure(matmul, "output must be a memref");

    std::optional<std::pair<bool, bool>> transposes = readTransposes(matmul);
    if (!transposes)
      return matmul.emitOpError()
             << "indexing_maps are not a plain or transposed matmul; the "
                "accelerator can only transpose an operand, not broadcast or "
                "permute one";
    auto trLhs = rewriter.getBoolAttr(transposes->first);
    auto trRhs = rewriter.getBoolAttr(transposes->second);
    auto dfAttr = DataflowAttr::get(rewriter.getContext(), dataflow);

    if (outType.getElementType().isInteger(32)) {
      // linalg.matmul means C += A*B. Skip reading C back only when it is
      // provably zero.
      linalg::FillOp zeroFill = zeroFillFor(out, matmul);
      bool accumulate = !zeroFill;
      // linalg.matmul is exact integer arithmetic, so the operands' mvin
      // scales stay at identity -- anything else would requantize the inputs.
      auto one = rewriter.getF32FloatAttr(1.0f);
      rewriter.create<MatMulInt8Op>(matmul.getLoc(), lhs, rhs, out,
                                    /*bias=*/Value(), one, one, trLhs, trRhs,
                                    rewriter.getBoolAttr(accumulate), dfAttr);
      eraseDeadZeroFill(rewriter, zeroFill);
    } else if (outType.getElementType().isInteger(8)) {
      // Tempting to send this to gemmlir.matmul_i8_scale, but the two do not
      // mean the same thing: linalg.matmul accumulates in the output element
      // type, so i8 x i8 -> i8 wraps, while the accelerator's scaled path
      // requantizes and *saturates*. There is also no scale to use. A quantized
      // matmul has to reach matmul_i8_scale from a pattern that says so.
      return matmul.emitOpError()
             << "i8 output cannot be offloaded: linalg.matmul wraps on overflow "
                "while the accelerator's scaled path saturates, and there is no "
                "scale to apply. Write gemmlir.matmul_i8_scale explicitly for a "
                "requantizing matmul";
    } else {
      return rewriter.notifyMatchFailure(matmul, "unsupported output element type");
    }

    rewriter.eraseOp(matmul);
    return success();
  }

private:
  Dataflow dataflow;
};

// linalg.matvec -> a matmul with one column.
//
// `y += A*x` is the N = 1 case of the runtime's matmul: the vector operands are
// reshaped to (K, 1) and (M, 1), whose row stride is 1, and the accelerator pads
// the short axis out to DIM itself.
class LinalgMatvecOpToGemmlir : public OpConversionPattern<linalg::MatvecOp> {
public:
  LinalgMatvecOpToGemmlir(MLIRContext *ctx, Dataflow dataflow)
      : OpConversionPattern(ctx), dataflow(dataflow) {}

  LogicalResult matchAndRewrite(linalg::MatvecOp matvec, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const final {
    ValueRange inputs = matvec.getInputs();
    ValueRange outputs = matvec.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return rewriter.notifyMatchFailure(matvec, "expected two inputs and one output");

    auto matTy = llvm::dyn_cast<MemRefType>(inputs[0].getType());
    auto vecTy = llvm::dyn_cast<MemRefType>(inputs[1].getType());
    auto outTy = llvm::dyn_cast<MemRefType>(outputs[0].getType());
    if (!matTy || !vecTy || !outTy || matTy.getRank() != 2 ||
        vecTy.getRank() != 1 || outTy.getRank() != 1)
      return rewriter.notifyMatchFailure(matvec, "expected memref<MxK>, memref<K> and memref<M>");
    if (!matTy.hasStaticShape() || !vecTy.hasStaticShape() || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(matvec, "operands must have a static shape");
    if (!matTy.getElementType().isInteger(8) || !vecTy.getElementType().isInteger(8))
      return rewriter.notifyMatchFailure(matvec, "inputs must be memref<...xi8>");
    if (!outTy.getElementType().isInteger(32))
      return matvec.emitOpError()
             << "only an i32 result can be offloaded: an i8 matvec would wrap "
                "where the accelerator's scaled path saturates";

    Location loc = matvec.getLoc();
    auto asColumn = [&](Value v, MemRefType t) -> Value {
      auto colTy = MemRefType::get({t.getShape()[0], 1}, t.getElementType());
      SmallVector<ReassociationIndices> reassoc = {{0, 1}};
      return rewriter.create<memref::ExpandShapeOp>(loc, colTy, v, reassoc);
    };

    linalg::FillOp zeroFill = zeroFillFor(outputs[0], matvec);
    bool accumulate = !zeroFill;
    auto no = rewriter.getBoolAttr(false);
    auto one = rewriter.getF32FloatAttr(1.0f);
    rewriter.create<MatMulInt8Op>(
        loc, inputs[0], asColumn(inputs[1], vecTy), asColumn(outputs[0], outTy),
        /*bias=*/Value(), one, one, no, no, rewriter.getBoolAttr(accumulate),
        DataflowAttr::get(rewriter.getContext(), dataflow));
    eraseDeadZeroFill(rewriter, zeroFill);

    rewriter.eraseOp(matvec);
    return success();
  }

private:
  Dataflow dataflow;
};

// linalg.vecmat -> a matmul with one row.
//
// The mirror of matvec: `y += x*A` is M = 1, so the vectors become
// single-*row* matrices, whose row stride is their own length.
class LinalgVecmatOpToGemmlir : public OpConversionPattern<linalg::VecmatOp> {
public:
  LinalgVecmatOpToGemmlir(MLIRContext *ctx, Dataflow dataflow)
      : OpConversionPattern(ctx), dataflow(dataflow) {}

  LogicalResult matchAndRewrite(linalg::VecmatOp vecmat, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const final {
    ValueRange inputs = vecmat.getInputs();
    ValueRange outputs = vecmat.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return rewriter.notifyMatchFailure(vecmat, "expected two inputs and one output");

    auto vecTy = llvm::dyn_cast<MemRefType>(inputs[0].getType());
    auto matTy = llvm::dyn_cast<MemRefType>(inputs[1].getType());
    auto outTy = llvm::dyn_cast<MemRefType>(outputs[0].getType());
    if (!vecTy || !matTy || !outTy || vecTy.getRank() != 1 ||
        matTy.getRank() != 2 || outTy.getRank() != 1)
      return rewriter.notifyMatchFailure(vecmat, "expected memref<K>, memref<KxN> and memref<N>");
    if (!vecTy.hasStaticShape() || !matTy.hasStaticShape() || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(vecmat, "operands must have a static shape");
    if (!vecTy.getElementType().isInteger(8) || !matTy.getElementType().isInteger(8))
      return rewriter.notifyMatchFailure(vecmat, "inputs must be memref<...xi8>");
    if (!outTy.getElementType().isInteger(32))
      return vecmat.emitOpError()
             << "only an i32 result can be offloaded: an i8 vecmat would wrap "
                "where the accelerator's scaled path saturates";

    Location loc = vecmat.getLoc();
    auto asRow = [&](Value v, MemRefType t) -> Value {
      auto rowTy = MemRefType::get({1, t.getShape()[0]}, t.getElementType());
      SmallVector<ReassociationIndices> reassoc = {{0, 1}};
      return rewriter.create<memref::ExpandShapeOp>(loc, rowTy, v, reassoc);
    };

    linalg::FillOp zeroFill = zeroFillFor(outputs[0], vecmat);
    bool accumulate = !zeroFill;
    auto no = rewriter.getBoolAttr(false);
    auto one = rewriter.getF32FloatAttr(1.0f);
    rewriter.create<MatMulInt8Op>(
        loc, asRow(inputs[0], vecTy), inputs[1], asRow(outputs[0], outTy),
        /*bias=*/Value(), one, one, no, no, rewriter.getBoolAttr(accumulate),
        DataflowAttr::get(rewriter.getContext(), dataflow));
    eraseDeadZeroFill(rewriter, zeroFill);

    rewriter.eraseOp(vecmat);
    return success();
  }

private:
  Dataflow dataflow;
};

// linalg.batch_matmul -> a loop of 2-D gemmlir.matmul_i8 over the batch.
//
// The accelerator's matmul is two-dimensional, so the batch axis is peeled into
// an scf.for whose body takes a rank-reduced memref.subview of each operand.
// Those subviews carry a dynamic offset, which the lowering folds into the
// pointer it hands the runtime.
class LinalgBatchMatmulOpToGemmlir
    : public OpConversionPattern<linalg::BatchMatmulOp> {
public:
  LinalgBatchMatmulOpToGemmlir(MLIRContext *ctx, Dataflow dataflow)
      : OpConversionPattern(ctx), dataflow(dataflow) {}

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp batch, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const final {
    ValueRange inputs = batch.getInputs();
    ValueRange outputs = batch.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return rewriter.notifyMatchFailure(batch, "expected two inputs and one output");

    SmallVector<MemRefType> types;
    for (Value v : {inputs[0], inputs[1], outputs[0]}) {
      auto mt = llvm::dyn_cast<MemRefType>(v.getType());
      if (!mt || mt.getRank() != 3 || !mt.hasStaticShape())
        return rewriter.notifyMatchFailure(batch, "operands must be 3-D static memrefs");
      types.push_back(mt);
    }
    if (!types[0].getElementType().isInteger(8) ||
        !types[1].getElementType().isInteger(8))
      return rewriter.notifyMatchFailure(batch, "inputs must be memref<...xi8>");
    if (!types[2].getElementType().isInteger(32))
      return batch.emitOpError()
             << "only an i32 result can be offloaded: an i8 batch_matmul would "
                "wrap where the accelerator's scaled path saturates";

    int64_t nbatch = types[2].getShape()[0];
    if (types[0].getShape()[0] != nbatch || types[1].getShape()[0] != nbatch)
      return batch.emitOpError("operands disagree on the batch size");

    Location loc = batch.getLoc();
    // Each slice is written once per iteration, so a zero fill outside the loop
    // still proves there is nothing to accumulate.
    linalg::FillOp zeroFill = zeroFillFor(outputs[0], batch);
    bool accumulate = !zeroFill;

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value ub = rewriter.create<arith::ConstantIndexOp>(loc, nbatch);
    auto loop = rewriter.create<scf::ForOp>(loc, zero, ub, one);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();

    auto slice = [&](Value v, MemRefType t) -> Value {
      SmallVector<OpFoldResult> offs = {iv, rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(t.getShape()[1]),
                                         rewriter.getIndexAttr(t.getShape()[2])};
      SmallVector<OpFoldResult> strides(3, rewriter.getIndexAttr(1));
      auto resTy = llvm::cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          {t.getShape()[1], t.getShape()[2]}, t, offs, sizes, strides));
      return rewriter.create<memref::SubViewOp>(loc, resTy, v, offs, sizes, strides);
    };

    auto no = rewriter.getBoolAttr(false);
    auto oneF32 = rewriter.getF32FloatAttr(1.0f);
    rewriter.create<MatMulInt8Op>(
        loc, slice(inputs[0], types[0]), slice(inputs[1], types[1]),
        slice(outputs[0], types[2]), /*bias=*/Value(), oneF32, oneF32, no, no,
        rewriter.getBoolAttr(accumulate),
        DataflowAttr::get(rewriter.getContext(), dataflow));
    eraseDeadZeroFill(rewriter, zeroFill);

    rewriter.eraseOp(batch);
    return success();
  }

private:
  Dataflow dataflow;
};

/// Matches `clamp(v, lo, hi)` written either way round, returning the clamped
/// value along with the bounds actually used.
static std::optional<Value> matchClamp(Value v, int64_t *lo, int64_t *hi) {
  auto outer = v.getDefiningOp();
  if (!outer)
    return std::nullopt;

  auto constOperand = [](Operation *op, Value *other) -> std::optional<int64_t> {
    APInt c;
    if (matchPattern(op->getOperand(1), m_ConstantInt(&c))) {
      *other = op->getOperand(0);
      return c.getSExtValue();
    }
    if (matchPattern(op->getOperand(0), m_ConstantInt(&c))) {
      *other = op->getOperand(1);
      return c.getSExtValue();
    }
    return std::nullopt;
  };

  Value mid;
  if (auto max = llvm::dyn_cast<arith::MaxSIOp>(outer)) {
    // max(min(v, hi), lo)
    std::optional<int64_t> l = constOperand(max, &mid);
    auto min = mid.getDefiningOp<arith::MinSIOp>();
    if (!l || !min)
      return std::nullopt;
    Value inner;
    std::optional<int64_t> h = constOperand(min, &inner);
    if (!h)
      return std::nullopt;
    *lo = *l; *hi = *h;
    return inner;
  }
  if (auto min = llvm::dyn_cast<arith::MinSIOp>(outer)) {
    // min(max(v, lo), hi)
    std::optional<int64_t> h = constOperand(min, &mid);
    auto max = mid.getDefiningOp<arith::MaxSIOp>();
    if (!h || !max)
      return std::nullopt;
    Value inner;
    std::optional<int64_t> l = constOperand(max, &inner);
    if (!l)
      return std::nullopt;
    *lo = *l; *hi = *h;
    return inner;
  }
  return std::nullopt;
}

/// Recognises a saturating i8 add written out in arith:
/// `trunci(clamp(addi(extsi a, extsi b), lo, 127))`, with `lo` either -128 or,
/// for a fused relu, 0. That is exactly what `tiled_resadd_auto` computes with
/// unit scales, and it is the shape a quantized model lowers a residual add to.
/// Plain `linalg.add` on i8 is *not* this -- it wraps.
static bool isSaturatingI8Add(linalg::GenericOp generic, bool *relu) {
  if (generic.getInputs().size() != 2 || generic.getOutputs().size() != 1)
    return false;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;
  if (!llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap m) { return m.isIdentity(); }))
    return false;

  for (Value v : llvm::concat<Value>(SmallVector<Value>(generic.getInputs()),
                                     SmallVector<Value>(generic.getOutputs()))) {
    auto mt = llvm::dyn_cast<MemRefType>(v.getType());
    if (!mt || mt.getRank() != 2 || !mt.hasStaticShape() ||
        !mt.getElementType().isInteger(8))
      return false;
  }
  auto shape = llvm::cast<MemRefType>(generic.getOutputs()[0].getType()).getShape();
  for (Value v : generic.getInputs())
    if (llvm::cast<MemRefType>(v.getType()).getShape() != shape)
      return false;

  Block &body = generic.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto trunc = yield.getOperand(0).getDefiningOp<arith::TruncIOp>();
  if (!trunc || !trunc.getType().isInteger(8))
    return false;

  int64_t lo = 0, hi = 0;
  std::optional<Value> sum = matchClamp(trunc.getIn(), &lo, &hi);
  if (!sum || hi != 127 || (lo != -128 && lo != 0))
    return false;

  auto add = sum->getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  auto lhsExt = add.getLhs().getDefiningOp<arith::ExtSIOp>();
  auto rhsExt = add.getRhs().getDefiningOp<arith::ExtSIOp>();
  if (!lhsExt || !rhsExt)
    return false;
  Value a = body.getArgument(0), b = body.getArgument(1);
  if (!((lhsExt.getIn() == a && rhsExt.getIn() == b) ||
        (lhsExt.getIn() == b && rhsExt.getIn() == a)))
    return false;

  *relu = (lo == 0);
  return true;
}

// Tried and refuted on the board: a **minimum size** for an accelerator call.
//
// The reasoning was that `tiled_resadd` ends in a fence and a sequential model
// has nothing to overlap it with, so a small enough add should stay on the core.
// Stubbing the LSTM's 32 gate-sum calls -- 1x192 each -- saves 7.75 ms of a
// 17.65 ms inference, which looked like the call paying for nothing.
//
// It is not. Leaving them on the core instead is **21.11 ms against 17.67**, at
// either floor tried (256 and 1024 elements). Even at 192 elements the array
// beats the loop; what the stub removed was the arithmetic, not an overhead the
// core could do more cheaply.

// A saturating i8 add -> gemmlir.resadd_i8.
class LinalgSaturatingAddToGemmlir : public OpConversionPattern<linalg::GenericOp> {
public:
  LinalgSaturatingAddToGemmlir(MLIRContext *ctx, Dataflow dataflow)
      : OpConversionPattern(ctx), dataflow(dataflow) {}

  LogicalResult matchAndRewrite(linalg::GenericOp generic, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const final {
    bool relu = false;
    if (!isSaturatingI8Add(generic, &relu))
      return rewriter.notifyMatchFailure(generic, "not a saturating i8 add");

    auto one = rewriter.getF32FloatAttr(1.0f);
    rewriter.create<ResAddInt8Op>(
        generic.getLoc(), generic.getInputs()[0], generic.getInputs()[1],
        generic.getOutputs()[0], one, one, one,
        ActAttr::get(rewriter.getContext(), relu ? Act::RELU : Act::NONE),
        DataflowAttr::get(rewriter.getContext(), dataflow));
    rewriter.eraseOp(generic);
    return success();
  }

private:
  Dataflow dataflow;
};

/// Matches an in-place `+= bias` over `out`, returning the bias operand.
///
/// Two shapes are accepted: a full i32 matrix added elementwise, and a row
/// broadcast down the matrix -- which is what the runtime's `repeating_bias`
/// does. `*broadcast` says which was found.
static std::optional<Value> matchBiasAdd(Operation *op, Value out,
                                         bool *broadcast) {
  auto generic = llvm::dyn_cast<linalg::GenericOp>(op);
  if (!generic || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1)
    return std::nullopt;
  if (generic.getOutputs()[0] != out)
    return std::nullopt;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return std::nullopt;

  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[1].isIdentity() || maps[1].getNumDims() != 2)
    return std::nullopt;

  MLIRContext *ctx = op->getContext();
  AffineExpr d0, d1;
  bindDims(ctx, d0, d1);
  if (maps[0] == maps[1])
    *broadcast = false;
  else if (maps[0] == AffineMap::get(2, 0, {d1}, ctx))
    *broadcast = true;
  else
    return std::nullopt;

  Block &body = generic.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  auto add = yield.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;
  Value in = body.getArgument(0), acc = body.getArgument(1);
  if (!((add.getLhs() == in && add.getRhs() == acc) ||
        (add.getLhs() == acc && add.getRhs() == in)))
    return std::nullopt;

  return generic.getInputs()[0];
}

// Folds a `+= bias` sitting on a matmul into the runtime's D operand.
//
// Sound on this path because nothing saturates: `arith.addi` wraps in i32 and
// so does the accelerator's 32-bit accumulator, whose raw value full_C reads
// back. Only applies when the matmul is not already accumulating -- there is
// one D pointer, and `accumulate` would be using it.
class FoldBiasIntoMatmul : public OpRewritePattern<MatMulInt8Op> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulInt8Op op,
                                PatternRewriter &rewriter) const final {
    if (op.getBias() || op.getAccumulate())
      return failure();
    Operation *next = nextUseOf(op, op.getOutMat());
    if (!next)
      return failure();

    bool broadcast = false;
    std::optional<Value> bias = matchBiasAdd(next, op.getOutMat(), &broadcast);
    if (!bias)
      return failure();

    auto biasTy = llvm::cast<MemRefType>(bias->getType());
    if (!biasTy.hasStaticShape() || !biasTy.getElementType().isInteger(32))
      return failure();

    Value biasVal = *bias;
    if (broadcast) {
      // The op wants a 2-D bias; a broadcast one is a single row.
      if (biasTy.getRank() != 1)
        return failure();
      auto rowTy = MemRefType::get({1, biasTy.getShape()[0]},
                                   biasTy.getElementType());
      SmallVector<ReassociationIndices> reassoc = {{0, 1}};
      rewriter.setInsertionPoint(op);
      biasVal = rewriter.create<memref::ExpandShapeOp>(op.getLoc(), rowTy,
                                                       biasVal, reassoc);
    } else if (biasTy.getRank() != 2) {
      return failure();
    }

    rewriter.eraseOp(next);
    rewriter.modifyOpInPlace(op, [&] { op.getBiasMutable().assign(biasVal); });
    return success();
  }
};

/// Matches `min(v, c)` in f32 for a constant `c`, written either way a frontend
/// spells it, returning the value being bounded.
static std::optional<Value> matchFloatUpperBound(Value v, double *bound) {
  auto constant = [&](Value c) {
    llvm::APFloat f(0.0f);
    if (!matchPattern(c, m_ConstantFloat(&f)))
      return false;
    *bound = f.convertToDouble();
    return true;
  };
  if (auto min = v.getDefiningOp<arith::MinimumFOp>()) {
    if (constant(min.getRhs()))
      return min.getLhs();
    if (constant(min.getLhs()))
      return min.getRhs();
    return std::nullopt;
  }
  if (auto min = v.getDefiningOp<arith::MinNumFOp>()) {
    if (constant(min.getRhs()))
      return min.getLhs();
    if (constant(min.getLhs()))
      return min.getRhs();
    return std::nullopt;
  }
  auto sel = v.getDefiningOp<arith::SelectOp>();
  if (!sel)
    return std::nullopt;
  auto cmp = sel.getCondition().getDefiningOp<arith::CmpFOp>();
  if (!cmp)
    return std::nullopt;
  bool lt = cmp.getPredicate() == arith::CmpFPredicate::OLT ||
            cmp.getPredicate() == arith::CmpFPredicate::ULT;
  bool gt = cmp.getPredicate() == arith::CmpFPredicate::OGT ||
            cmp.getPredicate() == arith::CmpFPredicate::UGT;
  // `a < b ? a : b` and `a > b ? b : a` are both the smaller of the two.
  Value small, large;
  if (lt && cmp.getLhs() == sel.getTrueValue() && cmp.getRhs() == sel.getFalseValue()) {
    small = sel.getTrueValue();
    large = sel.getFalseValue();
  } else if (gt && cmp.getRhs() == sel.getTrueValue() &&
             cmp.getLhs() == sel.getFalseValue()) {
    small = sel.getTrueValue();
    large = sel.getFalseValue();
  } else {
    return std::nullopt;
  }
  if (constant(small))
    return large;
  if (constant(large))
    return small;
  return std::nullopt;
}

/// Matches `max(v, c)` in f32 for a constant `c`, the mirror of the above.
static std::optional<Value> matchFloatLowerBound(Value v, double *bound) {
  auto constant = [&](Value c) {
    llvm::APFloat f(0.0f);
    if (!matchPattern(c, m_ConstantFloat(&f)))
      return false;
    *bound = f.convertToDouble();
    return true;
  };
  if (auto max = v.getDefiningOp<arith::MaximumFOp>()) {
    if (constant(max.getRhs()))
      return max.getLhs();
    if (constant(max.getLhs()))
      return max.getRhs();
    return std::nullopt;
  }
  if (auto max = v.getDefiningOp<arith::MaxNumFOp>()) {
    if (constant(max.getRhs()))
      return max.getLhs();
    if (constant(max.getLhs()))
      return max.getRhs();
    return std::nullopt;
  }
  auto sel = v.getDefiningOp<arith::SelectOp>();
  if (!sel)
    return std::nullopt;
  auto cmp = sel.getCondition().getDefiningOp<arith::CmpFOp>();
  if (!cmp)
    return std::nullopt;
  bool gt = cmp.getPredicate() == arith::CmpFPredicate::OGT ||
            cmp.getPredicate() == arith::CmpFPredicate::UGT;
  bool lt = cmp.getPredicate() == arith::CmpFPredicate::OLT ||
            cmp.getPredicate() == arith::CmpFPredicate::ULT;
  // `a > b ? a : b` and `a < b ? b : a` are both the larger of the two.
  Value larger, other;
  if (gt && cmp.getLhs() == sel.getTrueValue() && cmp.getRhs() == sel.getFalseValue()) {
    larger = sel.getTrueValue();
    other = sel.getFalseValue();
  } else if (lt && cmp.getRhs() == sel.getTrueValue() &&
             cmp.getLhs() == sel.getFalseValue()) {
    larger = sel.getTrueValue();
    other = sel.getFalseValue();
  } else {
    return std::nullopt;
  }
  if (constant(larger))
    return other;
  if (constant(other))
    return larger;
  return std::nullopt;
}

/// Matches `max(v, 0)` in f32, written either way a frontend spells it.
static std::optional<Value> matchFloatRelu(Value v) {
  auto isZero = [](Value c) { return matchPattern(c, m_AnyZeroFloat()); };
  if (auto max = v.getDefiningOp<arith::MaximumFOp>()) {
    if (isZero(max.getRhs()))
      return max.getLhs();
    if (isZero(max.getLhs()))
      return max.getRhs();
    return std::nullopt;
  }
  if (auto max = v.getDefiningOp<arith::MaxNumFOp>()) {
    if (isZero(max.getRhs()))
      return max.getLhs();
    if (isZero(max.getLhs()))
      return max.getRhs();
    return std::nullopt;
  }
  auto sel = v.getDefiningOp<arith::SelectOp>();
  if (!sel)
    return std::nullopt;
  auto cmp = sel.getCondition().getDefiningOp<arith::CmpFOp>();
  if (!cmp)
    return std::nullopt;
  bool gt = cmp.getPredicate() == arith::CmpFPredicate::UGT ||
            cmp.getPredicate() == arith::CmpFPredicate::OGT ||
            cmp.getPredicate() == arith::CmpFPredicate::UGE ||
            cmp.getPredicate() == arith::CmpFPredicate::OGE;
  if (!gt || !isZero(cmp.getRhs()) || sel.getTrueValue() != cmp.getLhs() ||
      !isZero(sel.getFalseValue()))
    return std::nullopt;
  return cmp.getLhs();
}

/// Matches a requantization of `acc` down to i8, returning the scale.
///
/// The core shape is `trunci(clamp(fptosi(roundeven(mulf(sitofp acc, s))), lo,
/// 127))`, `lo` being -128 or, for a fused relu, 0. The `roundeven` is not
/// optional: the accelerator's mvout scaling rounds half to even -- measured on
/// hardware, where it and truncation disagree on every exact .5 -- so a
/// requantize that truncates means something else and is left alone.
///
/// Three things a frontend adds are accepted on top, because each of them is
/// something the mvout pipeline does anyway:
///
///   * an i32 bias added to the accumulator first, which becomes the `D`
///     operand (`--quantize-bias-into-accumulator` puts it there);
///   * the scaling split into a dequantize and a requantize, `mulf s1` then
///     `divf s2` -- collapsed to the single `s1/s2` the hardware multiplies by,
///     which is the only form it has;
///   * a relu in f32 somewhere in that chain. Positive scaling is monotonic so
///     it commutes with both scalings, and `round(max(x,0)) == max(round(x),0)`,
///     so hoisting it out to `lo = 0` is exact.
static std::optional<llvm::APFloat>
matchRequantize(linalg::GenericOp generic, Value acc, bool *relu, Value *bias) {
  *bias = nullptr;
  size_t nIn = generic.getInputs().size();
  if ((nIn != 1 && nIn != 2) || generic.getOutputs().size() != 1)
    return std::nullopt;
  if (generic.getInputs()[0] != acc)
    return std::nullopt;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return std::nullopt;
  auto accTy = llvm::dyn_cast<MemRefType>(acc.getType());
  SmallVector<AffineMap> allMaps = generic.getIndexingMapsArray();
  if (!accTy || !accTy.hasStaticShape() || allMaps.size() != nIn + 1 ||
      !isWholeShapeRead(allMaps[0], accTy.getShape()) ||
      !allMaps.back().isIdentity())
    return std::nullopt;
  if (nIn == 2) {
    // Only the per-column broadcast the runtime can repeat.
    unsigned rank = allMaps[0].getNumDims();
    MLIRContext *ctx = generic.getContext();
    if (allMaps[1] !=
        AffineMap::get(rank, 0, {getAffineDimExpr(rank - 1, ctx)}, ctx))
      return std::nullopt;
    auto biasTy = llvm::dyn_cast<MemRefType>(generic.getInputs()[1].getType());
    if (!biasTy || !biasTy.hasStaticShape() ||
        !biasTy.getElementType().isInteger(32) || biasTy.getRank() != 1)
      return std::nullopt;
  }

  auto outTy = llvm::dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
  if (!outTy || !outTy.hasStaticShape() || !outTy.getElementType().isInteger(8))
    return std::nullopt;

  Block &body = generic.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  auto trunc = yield.getOperand(0).getDefiningOp<arith::TruncIOp>();
  if (!trunc || !trunc.getType().isInteger(8))
    return std::nullopt;

  int64_t lo = 0, hi = 0;
  std::optional<Value> clamped = matchClamp(trunc.getIn(), &lo, &hi);
  if (!clamped || hi != 127 || (lo != -128 && lo != 0))
    return std::nullopt;

  auto toInt = clamped->getDefiningOp<arith::FPToSIOp>();
  if (!toInt)
    return std::nullopt;
  auto round = toInt.getIn().getDefiningOp<math::RoundEvenOp>();
  if (!round)
    return std::nullopt;

  // Walk back through the scalings and the relu to the conversion.
  double scale = 1.0;
  bool floatRelu = false;
  Value v = round.getOperand();
  auto constant = [](Value c, double *out) {
    llvm::APFloat f(0.0f);
    if (!matchPattern(c, m_ConstantFloat(&f)))
      return false;
    *out = f.convertToDouble();
    return true;
  };
  while (true) {
    double c = 0.0;
    if (auto mul = v.getDefiningOp<arith::MulFOp>()) {
      if (constant(mul.getRhs(), &c))
        v = mul.getLhs();
      else if (constant(mul.getLhs(), &c))
        v = mul.getRhs();
      else
        return std::nullopt;
      if (!(c > 0.0))
        return std::nullopt;
      scale *= c;
      continue;
    }
    if (auto div = v.getDefiningOp<arith::DivFOp>()) {
      if (!constant(div.getRhs(), &c) || !(c > 0.0))
        return std::nullopt;
      scale /= c;
      v = div.getLhs();
      continue;
    }
    if (std::optional<Value> inner = matchFloatRelu(v)) {
      floatRelu = true;
      v = *inner;
      continue;
    }
    // A bounded activation -- ReLU6 is the one MobileNet is built from -- is
    // an upper clamp in f32 before the quantization. The accelerator has no
    // such activation, but it does not need one when the bound is already
    // outside what the quantization can represent: an activation calibrated at
    // `max|x| <= 6` has `6/scale >= 127`, so the saturation on the way out of
    // the accumulator clips first and the clamp never fires. Only then is it
    // dropped; a bound that bites is left where it is and the convolution does
    // not fold.
    double bound = 0.0;
    if (std::optional<Value> inner = matchFloatUpperBound(v, &bound)) {
      if (!(bound * scale >= 127.0))
        return std::nullopt;
      v = *inner;
      continue;
    }
    // And the lower half of a `hardtanh`. A relu is the case `bound == 0`,
    // which `matchFloatRelu` above has already taken; anything else is only
    // droppable when the saturation at -128 gets there first. With a relu in
    // the chain the narrowing clamps at 0 instead, and a negative bound can
    // never bite.
    if (std::optional<Value> inner = matchFloatLowerBound(v, &bound)) {
      if (!(floatRelu && bound <= 0.0) && !(bound * scale <= -128.0))
        return std::nullopt;
      v = *inner;
      continue;
    }
    // A bias of exactly zero. torchvision's SqueezeNet initialises every
    // convolution's bias to zero, so the per-channel tensor is a splat and the
    // frontend leaves `x + 0.0` behind -- twenty-six of them, each ending its
    // convolution's tail where the walk could not follow, which is why not one
    // of that model's convolutions reached the accelerator. `x + 0` differs
    // from `x` only in the sign of a zero, and both convert to the integer 0.
    // A bias that is not zero is a real one and stops the walk as before.
    if (auto add = v.getDefiningOp<arith::AddFOp>()) {
      if (constant(add.getRhs(), &c) && c == 0.0) {
        v = add.getLhs();
        continue;
      }
      if (constant(add.getLhs(), &c) && c == 0.0) {
        v = add.getRhs();
        continue;
      }
    }
    if (auto sub = v.getDefiningOp<arith::SubFOp>()) {
      if (constant(sub.getRhs(), &c) && c == 0.0) {
        v = sub.getLhs();
        continue;
      }
    }
    break;
  }
  if (!(scale > 0.0) || !std::isfinite(scale))
    return std::nullopt;

  auto toFloat = v.getDefiningOp<arith::SIToFPOp>();
  if (!toFloat)
    return std::nullopt;
  Value accVal = toFloat.getIn();
  if (nIn == 2) {
    auto add = accVal.getDefiningOp<arith::AddIOp>();
    if (!add)
      return std::nullopt;
    Value a = body.getArgument(0), b = body.getArgument(1);
    if (!((add.getLhs() == a && add.getRhs() == b) ||
          (add.getLhs() == b && add.getRhs() == a)))
      return std::nullopt;
    *bias = generic.getInputs()[1];
  } else if (accVal != body.getArgument(0)) {
    return std::nullopt;
  }

  *relu = (lo == 0) || floatRelu;
  return llvm::APFloat(static_cast<float>(scale));
}

// Folds a requantization onto a matmul, turning it into the scaled op that does
// the same thing in the mvout pipeline.
class FoldRequantizeIntoMatmul : public OpRewritePattern<MatMulInt8Op> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulInt8Op op,
                                PatternRewriter &rewriter) const final {
    if (op.getAccumulate())
      return failure();
    Value acc = op.getOutMat();

    // The i32 accumulator has to be a local temporary that nothing else reads,
    // or folding it away would drop a result someone is still using.
    //
    // The matmul may hold it by a *reshape* of that temporary rather than the
    // temporary itself: an im2col contraction writes `576x16` into what the
    // rest of the model sees as `1x24x24x16`, and the requantization reads the
    // four-dimensional view. A reshape names the same bytes in the same order,
    // so the two views are the same memory -- without this the whole tail of
    // every packed convolution stayed a scalar loop.
    Value base = reshapeBaseOf(acc);
    if (!base.getDefiningOp<memref::AllocOp>())
      return failure();

    Operation *next = nextUseOfBuffer(op, base);
    auto generic = llvm::dyn_cast_or_null<linalg::GenericOp>(next);
    if (!generic || generic.getInputs().empty() ||
        reshapeBaseOf(generic.getInputs()[0]) != base)
      return failure();
    // The scaled op replaces the matmul where the requantize is, so the
    // operands it reads must not have been written in between.
    if (!untouchedBetween(op, generic, {op.getLhsMat(), op.getRhsMat()}))
      return failure();

    bool relu = false;
    Value biasVal;
    std::optional<llvm::APFloat> scale =
        matchRequantize(generic, generic.getInputs()[0], &relu, &biasVal);
    if (!scale)
      return failure();

    // The result goes back through the same reshape. The runtime sees a matrix
    // and a row stride, so any split of a contiguous shape would do -- but this
    // one has to be the split the matmul already wrote, or the two would
    // disagree about where the rows are.
    Value out = generic.getOutputs()[0];
    auto outTy = llvm::cast<MemRefType>(out.getType());
    auto accTy = llvm::cast<MemRefType>(acc.getType());
    if (accTy.getRank() != 2)
      return failure();
    if (outTy.getRank() != 2) {
      if (!outTy.getLayout().isIdentity())
        return failure();
      SmallVector<ReassociationIndices> groups = asMatrix(outTy.getShape());
      SmallVector<int64_t> flat;
      for (ReassociationIndices &g : groups) {
        int64_t n = 1;
        for (int64_t d : g)
          n *= outTy.getShape()[d];
        flat.push_back(n);
      }
      if (flat.size() != 2 || flat[0] != accTy.getShape()[0] ||
          flat[1] != accTy.getShape()[1])
        return failure();
      rewriter.setInsertionPoint(generic);
      out = rewriter.create<memref::CollapseShapeOp>(op.getLoc(), out, groups);
    }
    // The op takes a 2-D bias; a per-column one is a single repeated row.
    if (biasVal) {
      if (op.getBias())
        return failure();
      auto biasTy = llvm::cast<MemRefType>(biasVal.getType());
      auto rowTy = MemRefType::get({1, biasTy.getShape()[0]},
                                   biasTy.getElementType());
      SmallVector<ReassociationIndices> reassoc = {{0, 1}};
      rewriter.setInsertionPoint(generic);
      biasVal = rewriter.create<memref::ExpandShapeOp>(op.getLoc(), rowTy,
                                                       biasVal, reassoc);
    } else {
      biasVal = op.getBias();
    }

    // Freeing the temporary afterwards is expected; reading it is not -- by
    // any of its views.
    for (Operation *later = generic->getNextNode(); later;
         later = later->getNextNode()) {
      if (llvm::isa<memref::DeallocOp, memref::CollapseShapeOp,
                    memref::ExpandShapeOp>(later))
        continue;
      for (Value v : later->getOperands())
        if (reshapeBaseOf(v) == base)
          return failure();
    }

    rewriter.setInsertionPoint(generic);
    rewriter.create<MatMulInt8ScaleOp>(
        op.getLoc(), op.getLhsMat(), op.getRhsMat(), out, biasVal, op.getLhsScaleAttr(), op.getRhsScaleAttr(),
        op.getTransposeLhsAttr(), op.getTransposeRhsAttr(),
        rewriter.getF32FloatAttr(scale->convertToFloat()),
        // Nothing here asks for an I-BERT activation, so the accumulator's
        // worth never gets read; leave it at the default.
        rewriter.getF32FloatAttr(1.0f),
        ActAttr::get(rewriter.getContext(), relu ? Act::RELU : Act::NONE),
        op.getDataflowAttr());
    rewriter.eraseOp(generic);
    rewriter.eraseOp(op);
    return success();
  }
};

/// The same fold, for a batch matmul.
///
/// `linalg.batch_matmul` becomes an `scf.for` over rank-reduced slices, so its
/// accumulator is written a slice at a time and the requantization that follows
/// reads the whole of it from outside the loop. `FoldRequantizeIntoMatmul` looks
/// for a 2-D accumulator the call holds directly and finds neither, so a
/// ConvNeXt block's pointwise convolutions and a transformer's attention kept a
/// full f32 pass over an i32 buffer that the mvout would have done for free.
///
/// Sinking it into the loop is sound because every slice gets the *same*
/// treatment: one scale for the whole tensor, and a bias the runtime repeats
/// down the rows, which is already per-column and so the same for every slice.
class FoldRequantizeIntoBatchMatmul : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const final {
    // The body has to be the slices and one call and nothing else: anything
    // else in there could read the accumulator between the call and the fold.
    MatMulInt8Op matmul;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (llvm::isa<memref::SubViewOp>(&op))
        continue;
      auto mm = llvm::dyn_cast<MatMulInt8Op>(&op);
      if (!mm || matmul)
        return failure();
      matmul = mm;
    }
    if (!matmul || matmul.getAccumulate() || matmul.getBias())
      return failure();

    auto accSlice = matmul.getOutMat().getDefiningOp<memref::SubViewOp>();
    if (!accSlice)
      return failure();
    Value acc = accSlice.getSource();
    auto accTy = llvm::dyn_cast<MemRefType>(acc.getType());
    // A local temporary, or folding it away would drop a result someone still
    // reads.
    if (!acc.getDefiningOp<memref::AllocOp>() || !accTy ||
        accTy.getRank() != 3 || !accTy.hasStaticShape())
      return failure();

    Operation *next = nextUseOfBuffer(loop, acc);
    auto generic = llvm::dyn_cast_or_null<linalg::GenericOp>(next);
    if (!generic)
      return failure();
    bool relu = false;
    Value biasVal;
    std::optional<llvm::APFloat> scale =
        matchRequantize(generic, acc, &relu, &biasVal);
    if (!scale)
      return failure();

    Value out = generic.getOutputs()[0];
    auto outTy = llvm::dyn_cast<MemRefType>(out.getType());
    // The slices are taken at the same offsets as the accumulator's, so the
    // output has to be laid out the same way and be the same shape.
    if (!outTy || outTy.getRank() != 3 || !outTy.getLayout().isIdentity() ||
        outTy.getShape() != accTy.getShape())
      return failure();

    // The call takes the requantization's place, so nothing may have written
    // the operands it reads in between -- the buffers the slices come from.
    SmallVector<Value> reads;
    if (auto lhs = matmul.getLhsMat().getDefiningOp<memref::SubViewOp>())
      reads.push_back(lhs.getSource());
    if (auto rhs = matmul.getRhsMat().getDefiningOp<memref::SubViewOp>())
      reads.push_back(rhs.getSource());
    if (reads.size() != 2 || !untouchedBetween(loop, generic, reads))
      return failure();

    // Freeing the temporary afterwards is expected; reading it is not.
    for (Operation *later = generic->getNextNode(); later;
         later = later->getNextNode()) {
      if (llvm::isa<memref::DeallocOp, memref::CollapseShapeOp,
                    memref::ExpandShapeOp>(later))
        continue;
      for (Value v : later->getOperands())
        if (reshapeBaseOf(v) == acc)
          return failure();
    }

    // The slice is taken *inside* the loop, so the buffer has to exist before
    // it. Bufferization puts the output's allocation between the loop and the
    // requantization that uses it, which is after; an allocation reads nothing,
    // so moving it up is free and it is the only thing worth moving.
    if (Operation *def = out.getDefiningOp()) {
      if (def->getBlock() != loop->getBlock())
        return failure();
      if (!def->isBeforeInBlock(loop)) {
        if (!llvm::isa<memref::AllocOp>(def) || !def->getOperands().empty())
          return failure();
        rewriter.moveOpBefore(def, loop);
      }
    }

    Location loc = matmul.getLoc();
    // The op takes a 2-D bias; a per-column one is a single repeated row, and
    // it is the same row for every slice.
    if (biasVal) {
      auto biasTy = llvm::cast<MemRefType>(biasVal.getType());
      auto rowTy = MemRefType::get({1, biasTy.getShape()[0]},
                                   biasTy.getElementType());
      SmallVector<ReassociationIndices> reassoc = {{0, 1}};
      rewriter.setInsertionPoint(loop);
      biasVal = rewriter.create<memref::ExpandShapeOp>(loc, rowTy, biasVal,
                                                       reassoc);
    }

    rewriter.setInsertionPoint(matmul);
    SmallVector<OpFoldResult> offs = accSlice.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = accSlice.getMixedSizes();
    SmallVector<OpFoldResult> strides = accSlice.getMixedStrides();
    auto sliceTy = llvm::cast<MemRefType>(
        memref::SubViewOp::inferRankReducedResultType(
            {accTy.getShape()[1], accTy.getShape()[2]}, outTy, offs, sizes,
            strides));
    Value outSlice = rewriter.create<memref::SubViewOp>(loc, sliceTy, out, offs,
                                                        sizes, strides);
    rewriter.create<MatMulInt8ScaleOp>(
        loc, matmul.getLhsMat(), matmul.getRhsMat(), outSlice, biasVal,
        matmul.getLhsScaleAttr(), matmul.getRhsScaleAttr(),
        matmul.getTransposeLhsAttr(), matmul.getTransposeRhsAttr(),
        rewriter.getF32FloatAttr(scale->convertToFloat()),
        rewriter.getF32FloatAttr(1.0f),
        ActAttr::get(rewriter.getContext(), relu ? Act::RELU : Act::NONE),
        matmul.getDataflowAttr());
    rewriter.eraseOp(generic);
    rewriter.eraseOp(matmul);
    return success();
  }
};

/// Writes a constant zero over the whole of `buf`.
static bool isZeroFillOf(Operation *op, Value buf) {
  if (auto fill = llvm::dyn_cast_or_null<linalg::FillOp>(op))
    return fill.getOutputs().size() == 1 && fill.getOutputs()[0] == buf &&
           fill.getInputs().size() == 1 &&
           matchPattern(fill.getInputs()[0], m_Zero());
  // Bufferizing a `tensor.pad` leaves the fill as a `linalg.map` with no inputs
  // whose body yields the constant.
  auto map = llvm::dyn_cast_or_null<linalg::MapOp>(op);
  if (!map || map.getInputs().size() != 0 || map.getInit() != buf)
    return false;
  Block &body = map.getMapper().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yield && yield.getNumOperands() == 1 &&
         matchPattern(yield.getOperand(0), m_Zero());
}

/// Folds a convolution's zero padding into the runtime's own `padding`.
///
/// `tensor.pad` bufferizes into an allocation, a fill of the whole of it, and a
/// copy of the real input into the middle -- 972 zero stores and a 768-element
/// strided copy on the CNN, for an input of 768. `tiled_conv_auto` takes a
/// `padding` and does it while it reads the image, so all of that goes away and
/// the convolution reads the unpadded buffer directly.
///
/// The runtime has a single `padding` scalar, so only a padding that is equal
/// on all four sides of the two spatial axes and absent on batch and channel
/// can be folded; anything else stays where it is. The value has to be zero,
/// which is what a symmetric quantization makes of a padded zero.
template <typename ConvOp>
class FoldPaddingIntoConv : public OpRewritePattern<ConvOp> {
public:
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp conv,
                                PatternRewriter &rewriter) const final {
    if (conv.getPadding() != 0)
      return failure();
    Value padded = conv.getInput();
    auto alloc = padded.getDefiningOp<memref::AllocOp>();
    auto paddedTy = llvm::dyn_cast<MemRefType>(padded.getType());
    if (!alloc || !paddedTy || !paddedTy.hasStaticShape() ||
        paddedTy.getRank() != 4)
      return failure();

    // Between the allocation and the call, the buffer must be filled with zero
    // and then written exactly once, by a copy into a centred window.
    Operation *fill = nullptr;
    memref::CopyOp copy;
    memref::SubViewOp window;
    for (Operation *n = alloc->getNextNode(); n && n != conv.getOperation();
         n = n->getNextNode()) {
      bool touchesBuffer = llvm::is_contained(n->getOperands(), padded);
      if (auto sv = llvm::dyn_cast<memref::SubViewOp>(n)) {
        if (sv.getSource() != padded)
          continue;
        if (window || !sv->hasOneUse())
          return failure();
        window = sv;
        continue;
      }
      if (touchesBuffer) {
        if (fill || !isZeroFillOf(n, padded))
          return failure();
        fill = n;
        continue;
      }
      if (window && llvm::is_contained(n->getOperands(), window.getResult())) {
        auto c = llvm::dyn_cast<memref::CopyOp>(n);
        if (copy || !c || c.getTarget() != window.getResult())
          return failure();
        copy = c;
      }
    }
    if (!fill || !copy || !window)
      return failure();
    // Nothing may read the padded buffer afterwards either.
    for (Operation *n = conv->getNextNode(); n; n = n->getNextNode())
      if (llvm::is_contained(n->getOperands(), padded) &&
          !llvm::isa<memref::DeallocOp>(n))
        return failure();

    auto srcTy = llvm::dyn_cast<MemRefType>(copy.getSource().getType());
    if (!srcTy || !srcTy.hasStaticShape() || srcTy.getRank() != 4)
      return failure();
    if (!window.getSource().getType().getLayout().isIdentity())
      return failure();

    SmallVector<int64_t> offsets, sizes, strides;
    if (window.getStaticStrides().empty())
      return failure();
    for (int64_t v : window.getStaticOffsets()) offsets.push_back(v);
    for (int64_t v : window.getStaticSizes()) sizes.push_back(v);
    for (int64_t v : window.getStaticStrides()) strides.push_back(v);
    if (offsets.size() != 4 || sizes.size() != 4 || strides.size() != 4)
      return failure();
    for (unsigned d = 0; d < 4; d++)
      if (sizes[d] != srcTy.getShape()[d] ||
          ShapedType::isDynamic(offsets[d]) || ShapedType::isDynamic(strides[d]))
        return failure();

    // Batch and channel untouched; the two spatial axes padded equally on all
    // four sides, which is the only shape the runtime's scalar can express.
    // A *strided* window on those axes is not padding at all: it is the
    // zero-stuffing a transposed convolution is written as, which the runtime
    // does with `input_dilation` -- the same border, with the source spread out
    // inside it.
    ArrayRef<int64_t> full = paddedTy.getShape();
    int64_t pad = offsets[1];
    int64_t spread = strides[1];
    if (strides[0] != 1 || strides[3] != 1 || strides[2] != spread || spread < 1)
      return failure();
    if (offsets[0] != 0 || offsets[3] != 0 || full[0] != sizes[0] ||
        full[3] != sizes[3] || offsets[2] != pad || pad < 0)
      return failure();
    if (spread == 1 && pad == 0)
      return failure(); // nothing to fold: the window is the whole buffer
    for (unsigned d = 1; d <= 2; d++)
      if (full[d] != spread * (sizes[d] - 1) + 1 + 2 * pad)
        return failure();
    if (spread != 1) {
      // The runtime's accelerator path takes an input dilation of 2 and only
      // with a unit stride; the depthwise call has no such parameter at all.
      if constexpr (!std::is_same_v<ConvOp, Conv2DInt8Op>)
        return failure();
      else if (spread != 2 || conv.getStride() != 1 ||
               conv.getInputDilation() != 1)
        return failure();
    }

    // `tiled_conv_auto` exits on `kernel_dim <= padding`, against the
    // *undilated* kernel. A dilated convolution's shape-preserving padding
    // (`dilation*(K-1)/2`) is past that limit as soon as the rate exceeds 2, so
    // leave the border where it is: an explicit zero buffer and a convolution
    // with no padding of its own computes the same thing, and it is the form
    // this pattern started from.
    auto filterTy = llvm::cast<MemRefType>(conv.getFilter().getType());
    // (KH, KW, C, F) for the dense kernel, (C, KH, KW) for the depthwise one.
    int64_t kernel = std::is_same_v<ConvOp, Conv2DInt8Op> ? filterTy.getShape()[0]
                                                          : filterTy.getShape()[1];
    if (pad >= kernel)
      return failure();

    rewriter.modifyOpInPlace(conv, [&] {
      conv.getInputMutable().assign(copy.getSource());
      conv.setPadding(pad);
      if constexpr (std::is_same_v<ConvOp, Conv2DInt8Op>)
        conv.setInputDilation(spread);
    });
    rewriter.eraseOp(copy);
    rewriter.eraseOp(window);
    rewriter.eraseOp(fill);
    return success();
  }
};

/// True when `op` writes `buf` and does not read it. The accelerator operations
/// declare their effects on the operation rather than on each value, so their
/// one destination is named here instead.
static bool writesOnly(Operation *op, Value buf) {
  auto destination = [&]() -> Value {
    if (auto o = llvm::dyn_cast<MatMulInt8Op>(op))
      return o.getAccumulate() ? Value() : o.getOutMat();
    if (auto o = llvm::dyn_cast<MatMulInt8ScaleOp>(op))
      return o.getOutMat();
    if (auto o = llvm::dyn_cast<ResAddInt8Op>(op))
      return o.getOutMat();
    if (auto o = llvm::dyn_cast<Conv2DInt8Op>(op))
      return o.getOutput();
    if (auto o = llvm::dyn_cast<DepthwiseConv2DInt8Op>(op))
      return o.getOutput();
    return Value();
  }();
  if (destination) {
    if (destination != buf)
      return false;
    // and not also an input
    for (Value v : op->getOperands())
      if (v == buf && v != destination)
        return false;
    return llvm::count(op->getOperands(), buf) == 1;
  }
  auto effects = llvm::dyn_cast<MemoryEffectOpInterface>(op);
  return effects && effects.getEffectOnValue<MemoryEffects::Write>(buf) &&
         !effects.getEffectOnValue<MemoryEffects::Read>(buf);
}

/// `memref.copy` between two allocations where the source is written once and
/// read nowhere else: the writer can write the target instead.
///
/// `--materialize-pad-sources` puts one of these in on purpose, to keep a
/// convolution's result out of the padded buffer it feeds -- and once
/// `FoldPaddingIntoConv` has taken the padding away there is nothing left for
/// it to separate, so it goes again. Refusing while the target is still a
/// window is what keeps the two from undoing each other: the window is not an
/// allocation.
class ElideRedundantBufferCopy : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const final {
    Value src = copy.getSource(), dst = copy.getTarget();
    auto srcTy = llvm::dyn_cast<MemRefType>(src.getType());
    auto dstTy = llvm::dyn_cast<MemRefType>(dst.getType());
    if (!srcTy || srcTy != dstTy || !srcTy.hasStaticShape() ||
        !srcTy.getLayout().isIdentity())
      return failure();
    auto srcAlloc = src.getDefiningOp<memref::AllocOp>();
    auto dstAlloc = dst.getDefiningOp<memref::AllocOp>();
    if (!srcAlloc || !dstAlloc || srcAlloc == dstAlloc ||
        !srcAlloc.getDynamicSizes().empty() ||
        !dstAlloc.getDynamicSizes().empty() ||
        srcAlloc->getBlock() != copy->getBlock() ||
        dstAlloc->getBlock() != copy->getBlock())
      return failure();

    // One operation writes the source, before the copy, and nothing reads it.
    Operation *writer = nullptr;
    for (Operation *user : src.getUsers()) {
      if (user == copy || llvm::isa<memref::DeallocOp>(user))
        continue;
      if (writer || !user->isBeforeInBlock(copy))
        return failure();
      if (!writesOnly(user, src))
        return failure();
      writer = user;
    }
    if (!writer)
      return failure();
    // And nothing touches the target before the copy: it is the copy's value
    // from here on.
    for (Operation *user : dst.getUsers())
      if (user != copy && !llvm::isa<memref::DeallocOp>(user) &&
          user->isBeforeInBlock(copy))
        return failure();

    if (!dstAlloc->isBeforeInBlock(writer))
      rewriter.moveOpBefore(dstAlloc, writer);
    rewriter.modifyOpInPlace(writer, [&] {
      for (OpOperand &operand : writer->getOpOperands())
        if (operand.get() == src)
          operand.set(dst);
    });
    rewriter.eraseOp(copy);
    return success();
  }
};

/// A convolution whose result is then copied into one slice of a wider buffer
/// writes that slice itself.
///
/// `torch.cat` on the channel axis is what an Inception block, a DenseNet layer
/// and a detection neck are joined with, and it bufferizes into an allocation
/// per branch plus a strided copy into the join. The copies are not small: two
/// 2048-element strided i8 copies cost **4.61 ms of a 4.96 ms inference** on the
/// board, because a strided `memref.copy` goes through the runtime's
/// element-at-a-time walk. `tiled_conv_stride_auto` takes the distance between
/// two output pixels, so the convolution can write into the join directly and
/// the copy disappears.
/// The buffer a view ultimately refers to, so two views of one allocation are
/// recognised as the same memory.
static Value joinBaseOf(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto op = llvm::dyn_cast<memref::SubViewOp>(def)) { v = op.getSource(); continue; }
    if (auto op = llvm::dyn_cast<memref::ViewOp>(def)) { v = op.getSource(); continue; }
    if (auto op = llvm::dyn_cast<memref::CastOp>(def)) { v = op.getSource(); continue; }
    if (auto op = llvm::dyn_cast<memref::ExpandShapeOp>(def)) { v = op.getSrc(); continue; }
    if (auto op = llvm::dyn_cast<memref::CollapseShapeOp>(def)) { v = op.getSrc(); continue; }
    break;
  }
  return v;
}

/// Pointing a producer at one slice of a wider buffer moves its write earlier,
/// to where the producer is. Anything between that also touches the buffer has
/// to be dealt with, or it acts on the slice at the wrong time: bufferizing a
/// padded convolution puts the zero fill of the border *after* the convolution
/// that fills the middle, and redirecting without noticing lets the fill erase
/// what the convolution just wrote. That is a silent wrong answer -- measured,
/// a ResNeXt block at 0.1029 relative L2 where the same block ungrouped is
/// 0.0026, because the whole grouped branch was reading zeros.
///
/// A fill of the *whole* buffer is the one case that can be kept: running it
/// before the producer instead writes the same bytes, since the producer's
/// slice was going to be overwritten either way. Everything else refuses.
static LogicalResult clearThePathToTheJoin(PatternRewriter &rewriter,
                                           Operation *producer, Operation *copy,
                                           Value join) {
  Value base = joinBaseOf(join);
  SmallVector<Operation *> hoist;
  for (Operation *n = producer->getNextNode(); n && n != copy;
       n = n->getNextNode()) {
    if (isMemoryEffectFree(n))
      continue;
    bool touches = llvm::any_of(n->getOperands(), [&](Value v) {
      return llvm::isa<MemRefType>(v.getType()) && joinBaseOf(v) == base;
    });
    if (!touches)
      continue;
    // Writes every element of the buffer itself, out of values that are
    // already available where the producer is.
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(n);
    if (!linalgOp || linalgOp.getNumDpsInits() != 1 ||
        linalgOp.getDpsInits()[0] != base ||
        !llvm::all_of(linalgOp.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }) ||
        !linalgOp.getIndexingMapsArray().back().isIdentity())
      return failure();
    // Everything it reads has to be available up there -- and a linalg op does
    // not read only through its operands. `linalg.map` filling a buffer with a
    // scalar **captures** that scalar in its region, where `getOperands` cannot
    // see it: SqueezeNet's ceil-mode max-pool pads with one, and hoisting the
    // fill above the value it yields is how this produced IR that did not
    // verify.
    SmallVector<Value> reads(n->getOperands());
    SetVector<Value> captured;
    getUsedValuesDefinedAbove(n->getRegions(), captured);
    reads.append(captured.begin(), captured.end());
    for (Value v : reads)
      if (Operation *def = v.getDefiningOp()) {
        if (def->getBlock() != producer->getBlock() ||
            !def->isBeforeInBlock(producer))
          return failure();
      }
    hoist.push_back(n);
  }
  for (Operation *n : hoist)
    rewriter.moveOpBefore(n, producer);
  return success();
}

class FoldConcatIntoConv : public OpRewritePattern<Conv2DInt8Op> {
public:
  using OpRewritePattern<Conv2DInt8Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DInt8Op conv,
                                PatternRewriter &rewriter) const final {
    Value buf = conv.getOutput();
    auto bufTy = llvm::dyn_cast<MemRefType>(buf.getType());
    if (!buf.getDefiningOp<memref::AllocOp>() || !bufTy ||
        !bufTy.getLayout().isIdentity())
      return failure();

    // The only thing that may happen to the result is the copy into the join,
    // and nothing may read the buffer afterwards.
    memref::CopyOp copy;
    for (Operation *later = conv->getNextNode(); later;
         later = later->getNextNode()) {
      if (!llvm::is_contained(later->getOperands(), buf))
        continue;
      if (llvm::isa<memref::DeallocOp>(later))
        continue;
      auto c = llvm::dyn_cast<memref::CopyOp>(later);
      if (copy || !c || c.getSource() != buf)
        return failure();
      copy = c;
    }
    if (!copy)
      return failure();
    auto window = copy.getTarget().getDefiningOp<memref::SubViewOp>();
    if (!window || !window.getSource().getType().getLayout().isIdentity())
      return failure();

    auto joinTy = llvm::cast<MemRefType>(window.getSource().getType());
    auto sliceTy = llvm::cast<MemRefType>(window.getType());
    if (joinTy.getRank() != 4 || sliceTy.getRank() != 4)
      return failure();
    if (window.getStaticOffsets().size() != 4 ||
        window.getStaticSizes().size() != 4 ||
        window.getStaticStrides().size() != 4)
      return failure();
    for (int64_t v : window.getStaticStrides())
      if (v != 1)
        return failure();
    ArrayRef<int64_t> offsets = window.getStaticOffsets();
    ArrayRef<int64_t> sizes = window.getStaticSizes();
    // Batch and the two spatial axes whole, a window on the channels only:
    // that is a join, and the runtime's `out_stride` expresses exactly it.
    for (unsigned d = 0; d < 3; d++)
      if (offsets[d] != 0 || sizes[d] != joinTy.getDimSize(d) ||
          sizes[d] != bufTy.getDimSize(d))
        return failure();
    if (ShapedType::isDynamic(offsets[3]) || offsets[3] < 0 ||
        sizes[3] != bufTy.getDimSize(3) ||
        offsets[3] + sizes[3] > joinTy.getDimSize(3))
      return failure();

    // Bufferization puts the window and the copy after the convolution, so the
    // window has to be taken again where the convolution can see it. The buffer
    // being joined into has to be there already.
    // A block argument is there for the whole function; anything else has to be
    // reachable from the convolution.
    Operation *join = window.getSource().getDefiningOp();
    if (join && join->getBlock() != conv->getBlock())
      return failure();
    if (join && !join->isBeforeInBlock(conv)) {
      // Bufferization allocates the buffer being joined into only where the
      // first copy needs it, which is after both branches have run. An
      // allocation of a static shape has no operands, so it can be taken up to
      // where the branches are.
      auto alloc = llvm::dyn_cast<memref::AllocOp>(join);
      if (!alloc || !alloc.getDynamicSizes().empty() ||
          !alloc.getSymbolOperands().empty())
        return failure();
      rewriter.moveOpBefore(join, conv);
    }

    if (failed(clearThePathToTheJoin(rewriter, conv, copy, window.getSource())))
      return failure();

    rewriter.setInsertionPoint(conv);
    auto moved = rewriter.create<memref::SubViewOp>(
        window.getLoc(), window.getSource(), window.getMixedOffsets(),
        window.getMixedSizes(), window.getMixedStrides());
    rewriter.modifyOpInPlace(conv, [&] {
      conv.getOutputMutable().assign(moved.getResult());
    });
    rewriter.eraseOp(copy);
    return success();
  }
};

/// The same for a buffer an elementwise operation fills.
///
/// A branch of a concatenation is not always a convolution: ShuffleNet's unit
/// passes half the channels through, so the piece joined in is whatever
/// requantized them. The copy costs the same either way -- a strided
/// `memref.copy` goes through the runtime's element-at-a-time walk, 3.66 ms of
/// a 4.52 ms inference for one 2048-element slice -- and an operation that
/// writes every element of its destination can be pointed at the join instead.
class FoldConcatIntoElementwise : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const final {
    Value buf = copy.getSource();
    auto bufTy = llvm::dyn_cast<MemRefType>(buf.getType());
    if (!buf.getDefiningOp<memref::AllocOp>() || !bufTy ||
        !bufTy.getLayout().isIdentity())
      return failure();
    auto window = copy.getTarget().getDefiningOp<memref::SubViewOp>();
    if (!window || !window.getSource().getType().getLayout().isIdentity())
      return failure();
    for (int64_t v : window.getStaticStrides())
      if (v != 1)
        return failure();

    // One operation fills the buffer, and nothing wants it after the copy.
    linalg::GenericOp filler;
    for (Operation *user : buf.getDefiningOp()->getUsers()) {
      if (llvm::isa<memref::DeallocOp>(user) || user == copy.getOperation())
        continue;
      auto generic = llvm::dyn_cast<linalg::GenericOp>(user);
      if (filler || !generic || generic.getOutputs().size() != 1 ||
          generic.getOutputs()[0] != buf)
        return failure();
      filler = generic;
    }
    if (!filler || !filler->isBeforeInBlock(copy))
      return failure();
    for (Operation *later = copy->getNextNode(); later;
         later = later->getNextNode())
      if (llvm::is_contained(later->getOperands(), buf) &&
          !llvm::isa<memref::DeallocOp>(later))
        return failure();

    // Every element written, or what was in the join would show through.
    if (!llvm::all_of(filler.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return failure();
    if (!filler.getIndexingMapsArray().back().isIdentity())
      return failure();
    if (!filler.getRegion().front().getArguments().back().use_empty())
      return failure();
    // And the destination has to be reachable from where it is written. As
    // with a convolution, bufferization allocates the buffer being joined into
    // only where the first copy needs it; an allocation of a static shape has
    // no operands, so it can be taken up to where the branches are.
    Operation *join = window.getSource().getDefiningOp();
    if (join && join->getBlock() != filler->getBlock())
      return failure();
    if (join && !join->isBeforeInBlock(filler)) {
      auto alloc = llvm::dyn_cast<memref::AllocOp>(join);
      if (!alloc || !alloc.getDynamicSizes().empty() ||
          !alloc.getSymbolOperands().empty())
        return failure();
      rewriter.moveOpBefore(join, filler);
    }

    if (failed(clearThePathToTheJoin(rewriter, filler, copy, window.getSource())))
      return failure();

    rewriter.setInsertionPoint(filler);
    auto moved = rewriter.create<memref::SubViewOp>(
        window.getLoc(), window.getSource(), window.getMixedOffsets(),
        window.getMixedSizes(), window.getMixedStrides());
    rewriter.modifyOpInPlace(filler, [&] {
      filler.getOutputsMutable()[0].assign(moved.getResult());
    });
    rewriter.eraseOp(copy);
    return success();
  }
};

/// The buffer an accelerator operation writes *without reading it first*.
///
/// An accumulating matmul is the exception: `C += A*B` hands C over as the
/// bias, so a host write to it is the accumulator's starting value and is meant
/// to be there.
static Value overwrittenOutput(Operation *op) {
  if (auto m = llvm::dyn_cast<MatMulInt8Op>(op))
    return m.getAccumulate() ? Value() : m.getOutMat();
  if (auto m = llvm::dyn_cast<MatMulInt8ScaleOp>(op))
    return m.getOutMat();
  if (auto c = llvm::dyn_cast<Conv2DInt8Op>(op))
    return c.getOutput();
  if (auto c = llvm::dyn_cast<DepthwiseConv2DInt8Op>(op))
    return c.getOutput();
  return {};
}

/// True when `op` writes `buf`.
static bool writes(Operation *op, Value buf) {
  if (auto copy = llvm::dyn_cast<memref::CopyOp>(op))
    return copy.getTarget() == buf;
  if (auto store = llvm::dyn_cast<memref::StoreOp>(op))
    return store.getMemRef() == buf;
  if (auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op))
    return llvm::is_contained(linalgOp.getDpsInits(), buf);
  return false;
}

/// Rejects a host write to a buffer the accelerator overwrites without reading.
///
/// Gemmini's writes do not invalidate this board's data cache, so a line the
/// CPU wrote survives the accelerator overwriting the memory underneath it and
/// the CPU reads its own stale data afterwards. Measured directly: filling the
/// output buffer before the call left 43 to 69 of 784 elements reading back
/// wrong, six rounds running, and evicting the cache between the call and the
/// read -- or simply never touching the buffer -- gave 0 of 784, six rounds
/// running. There is no cache maintenance instruction to do it properly with:
/// the board's ISA is rv64imafdc, no Zicbom.
///
/// Only a write the accelerator then *overwrites without reading* is rejected,
/// because such a write is dead anyway -- the zero fills this pipeline erases
/// as it finds them, which turns out to be a correctness rule and not only an
/// optimization. An accumulating matmul reads its output as the bias, so a
/// write there is meant to be there and is left alone; the same hazard applies
/// to it, but the fix is the caller's (keep the buffer out of the cache), not
/// the compiler's.
static LogicalResult checkHostWrites(ModuleOp module) {
  StringRef dialect = GemmlirDialect::getDialectNamespace();
  WalkResult result = module.walk([&](Operation *op) {
    if (!op->getDialect() || op->getDialect()->getNamespace() != dialect)
      return WalkResult::advance();
    Value out = overwrittenOutput(op);
    if (!out)
      return WalkResult::advance();
    for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode()) {
      if (prev == out.getDefiningOp())
        break;
      if (!writes(prev, out))
        continue;
      op->emitError()
          << "the host writes this operation's output buffer beforehand; "
             "Gemmini's writes do not invalidate the data cache on this board, "
             "so the result would be read back stale. Remove the write -- a "
             "zero fill in front of a non-accumulating call is already dead";
      prev->emitRemark() << "the write is here";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

/// Matches `acc += bias` broadcast over the trailing (channel) axis.
static std::optional<Value> matchChannelBiasAdd(Operation *op, Value acc) {
  auto generic = llvm::dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1)
    return std::nullopt;
  if (generic.getOutputs()[0] != acc)
    return std::nullopt;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return std::nullopt;

  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[1].isIdentity())
    return std::nullopt;
  unsigned rank = maps[1].getNumDims();
  MLIRContext *ctx = op->getContext();
  if (maps[0] != AffineMap::get(rank, 0,
                                {getAffineDimExpr(rank - 1, ctx)}, ctx))
    return std::nullopt;

  Block &body = generic.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  auto add = yield.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;
  Value in = body.getArgument(0), out = body.getArgument(1);
  if (!((add.getLhs() == in && add.getRhs() == out) ||
        (add.getLhs() == out && add.getRhs() == in)))
    return std::nullopt;
  return generic.getInputs()[0];
}

// A quantized convolution -> gemmlir.conv2d_i8.
//
// `tiled_conv_auto` always writes elem_t, so a bare linalg conv accumulating
// into i32 has nothing to lower to: the whole chain has to be here --
// conv into a local i32 temporary, an optional per-channel bias, and a
// requantization down to i8. Padding is not part of linalg's conv (it is a
// separate pad on the input), so the offloaded call uses padding = 0.
class FuseQuantizedConv : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp conv,
                                PatternRewriter &rewriter) const final {
    if (conv.getInputs().size() != 2 || conv.getOutputs().size() != 1)
      return failure();
    Value input = conv.getInputs()[0], filter = conv.getInputs()[1];
    Value acc = conv.getOutputs()[0];

    auto inTy = llvm::dyn_cast<MemRefType>(input.getType());
    auto fTy = llvm::dyn_cast<MemRefType>(filter.getType());
    auto accTy = llvm::dyn_cast<MemRefType>(acc.getType());
    if (!inTy || !fTy || !accTy || inTy.getRank() != 4 || fTy.getRank() != 4 ||
        accTy.getRank() != 4)
      return failure();
    if (!inTy.hasStaticShape() || !fTy.hasStaticShape() || !accTy.hasStaticShape())
      return failure();
    if (!inTy.getElementType().isInteger(8) ||
        !fTy.getElementType().isInteger(8) ||
        !accTy.getElementType().isInteger(32))
      return failure();
    // linalg.conv accumulates into its output, and the runtime call does not
    // read that buffer at all, so the accumulator has to start at zero for the
    // two to mean the same thing. The fill that proves it is then dead -- the
    // call writes the requantized result to a different buffer -- and it is
    // 2832 stores an inference on the CNN.
    linalg::FillOp accZero = zeroFillFor(acc, conv);
    if (!accZero)
      return failure();
    if (fTy.getShape()[0] != fTy.getShape()[1])
      return failure(); // the runtime takes a single kernel_dim
    if (!acc.getDefiningOp<memref::AllocOp>())
      return failure();

    // One stride and one dilation, equal on both axes.
    auto pair = [](DenseIntElementsAttr a) -> std::optional<int64_t> {
      if (!a || a.getNumElements() != 2)
        return std::nullopt;
      auto it = a.value_begin<APInt>();
      int64_t x = (*it).getSExtValue(), y = (*(it + 1)).getSExtValue();
      return x == y ? std::optional<int64_t>(x) : std::nullopt;
    };
    std::optional<int64_t> stride = pair(conv.getStrides());
    std::optional<int64_t> dilation = pair(conv.getDilations());
    if (!stride || !dilation)
      return failure();

    // A channel bias reaches the runtime's D either as its own operation or
    // folded into the requantization, depending on how far the elementwise
    // fusion got; both are the same i32 vector.
    auto usableBias = [&](Value b) {
      auto bTy = llvm::dyn_cast<MemRefType>(b.getType());
      return bTy && bTy.getRank() == 1 && bTy.hasStaticShape() &&
             bTy.getElementType().isInteger(32) &&
             bTy.getShape()[0] == fTy.getShape()[3];
    };

    Operation *next = nextUseOf(conv, acc);
    Value bias;
    Operation *biasOp = nullptr;
    if (std::optional<Value> b = matchChannelBiasAdd(next, acc)) {
      if (!usableBias(*b))
        return failure();
      bias = *b;
      biasOp = next;
      next = nextUseOf(next, acc);
    }

    auto requant = llvm::dyn_cast_or_null<linalg::GenericOp>(next);
    if (!requant)
      return failure();
    bool relu = false;
    Value innerBias;
    std::optional<llvm::APFloat> scale =
        matchRequantize(requant, acc, &relu, &innerBias);
    if (!scale)
      return failure();
    if (innerBias) {
      if (bias || !usableBias(innerBias))
        return failure();
      bias = innerBias;
    }
    // The call takes the requantization's place, so nothing may have written
    // the operands it reads in between.
    if (!untouchedBetween(conv, requant, {input, filter}))
      return failure();

    auto outTy = llvm::cast<MemRefType>(requant.getOutputs()[0].getType());
    if (outTy.getRank() != 4 || outTy.getShape() != accTy.getShape())
      return failure();

    for (Operation *later = requant->getNextNode(); later;
         later = later->getNextNode())
      if (llvm::is_contained(later->getOperands(), acc) &&
          !llvm::isa<memref::DeallocOp>(later))
        return failure();

    if (!convBufferIsWalkable(input) ||
        !convBufferIsWalkable(requant.getOutputs()[0]))
      return failure();

    rewriter.setInsertionPoint(requant);
    rewriter.create<Conv2DInt8Op>(
        conv.getLoc(), input, filter, bias, requant.getOutputs()[0],
        rewriter.getI64IntegerAttr(*stride), rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(*dilation), rewriter.getI64IntegerAttr(1),
        rewriter.getF32FloatAttr(scale->convertToFloat()),
        ActAttr::get(rewriter.getContext(), relu ? Act::RELU : Act::NONE),
        rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(0),
        DataflowAttr::get(rewriter.getContext(), Dataflow::WS));

    rewriter.eraseOp(requant);
    if (biasOp)
      rewriter.eraseOp(biasOp);
    rewriter.eraseOp(conv);
    eraseDeadZeroFill(rewriter, accZero);
    return success();
  }
};

/// Re-lays a constant filter out, at compile time.
///
/// `linalg.depthwise_conv_2d_nhwc_hwc` reads (KH, KW, C); `tiled_conv_dw_auto`
/// reads (C, KH, KW). Both are constants here, so the permutation is a new
/// global rather than a loop.
static Value relayoutConstantFilter(PatternRewriter &rewriter, Operation *at,
                                    Value filter) {
  auto get = filter.getDefiningOp<memref::GetGlobalOp>();
  if (!get)
    return {};
  auto module = at->getParentOfType<ModuleOp>();
  auto global = llvm::dyn_cast_or_null<memref::GlobalOp>(
      SymbolTable::lookupSymbolIn(module, get.getNameAttr()));
  if (!global || !global.getConstant() || !global.getInitialValue())
    return {};
  auto dense = llvm::dyn_cast<DenseElementsAttr>(*global.getInitialValue());
  auto type = llvm::dyn_cast<MemRefType>(filter.getType());
  if (!dense || !type || type.getRank() != 3 || !type.hasStaticShape())
    return {};

  int64_t kh = type.getShape()[0], kw = type.getShape()[1],
          c = type.getShape()[2];
  SmallVector<Attribute> values(dense.getValues<Attribute>());
  SmallVector<Attribute> out;
  out.reserve(values.size());
  for (int64_t ci = 0; ci < c; ci++)
    for (int64_t i = 0; i < kh; i++)
      for (int64_t j = 0; j < kw; j++)
        out.push_back(values[(i * kw + j) * c + ci]);

  auto outTy = MemRefType::get({c, kh, kw}, type.getElementType());
  auto tensorTy = RankedTensorType::get({c, kh, kw}, type.getElementType());
  std::string name = (get.getName() + "_chw").str();
  if (!SymbolTable::lookupSymbolIn(module, name)) {
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPoint(global);
    builder.create<memref::GlobalOp>(
        global.getLoc(), builder.getStringAttr(name),
        builder.getStringAttr("private"), TypeAttr::get(outTy),
        DenseElementsAttr::get(tensorTy, out), /*constant=*/builder.getUnitAttr(),
        global.getAlignmentAttr());
  }
  return rewriter.create<memref::GetGlobalOp>(get.getLoc(), outTy, name);
}

/// The depthwise twin of FuseQuantizedConv.
///
/// Same chain -- an i8 convolution into an i32 accumulator, a per-channel bias
/// and a requantization -- with one filter dimension fewer, and the filter
/// re-laid-out because linalg counts it (KH, KW, C) where the runtime counts it
/// (C, KH, KW). torch-mlir emits this for a `groups == channels` convolution,
/// which is what a MobileNet-shaped model is made of.
class FuseQuantizedDepthwiseConv
    : public OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcOp conv,
                                PatternRewriter &rewriter) const final {
    if (conv.getInputs().size() != 2 || conv.getOutputs().size() != 1)
      return failure();
    Value input = conv.getInputs()[0], filter = conv.getInputs()[1];
    Value acc = conv.getOutputs()[0];

    auto inTy = llvm::dyn_cast<MemRefType>(input.getType());
    auto fTy = llvm::dyn_cast<MemRefType>(filter.getType());
    auto accTy = llvm::dyn_cast<MemRefType>(acc.getType());
    if (!inTy || !fTy || !accTy || inTy.getRank() != 4 || fTy.getRank() != 3 ||
        accTy.getRank() != 4 || !inTy.hasStaticShape() ||
        !fTy.hasStaticShape() || !accTy.hasStaticShape())
      return failure();
    if (!inTy.getElementType().isInteger(8) ||
        !fTy.getElementType().isInteger(8) ||
        !accTy.getElementType().isInteger(32))
      return failure();
    if (fTy.getShape()[0] != fTy.getShape()[1])
      return failure(); // the runtime takes a single kernel_dim
    int64_t channels = fTy.getShape()[2];
    if (inTy.getShape()[3] != channels || accTy.getShape()[3] != channels)
      return failure();
    if (!acc.getDefiningOp<memref::AllocOp>())
      return failure();
    linalg::FillOp accZero = zeroFillFor(acc, conv);
    if (!accZero)
      return failure();

    auto pair = [](DenseIntElementsAttr a) -> std::optional<int64_t> {
      if (!a || a.getNumElements() != 2)
        return std::nullopt;
      auto it = a.value_begin<APInt>();
      int64_t x = (*it).getSExtValue(), y = (*(it + 1)).getSExtValue();
      return x == y ? std::optional<int64_t>(x) : std::nullopt;
    };
    std::optional<int64_t> stride = pair(conv.getStrides());
    std::optional<int64_t> dilation = pair(conv.getDilations());
    if (!stride || !dilation || *dilation != 1)
      return failure(); // the runtime's depthwise path has no dilation

    auto usableBias = [&](Value b) {
      auto bTy = llvm::dyn_cast<MemRefType>(b.getType());
      return bTy && bTy.getRank() == 1 && bTy.hasStaticShape() &&
             bTy.getElementType().isInteger(32) &&
             bTy.getShape()[0] == channels;
    };

    Operation *next = nextUseOf(conv, acc);
    Value bias;
    Operation *biasOp = nullptr;
    if (std::optional<Value> b = matchChannelBiasAdd(next, acc)) {
      if (!usableBias(*b))
        return failure();
      bias = *b;
      biasOp = next;
      next = nextUseOf(next, acc);
    }
    auto requant = llvm::dyn_cast_or_null<linalg::GenericOp>(next);
    if (!requant)
      return failure();
    bool relu = false;
    Value innerBias;
    std::optional<llvm::APFloat> scale =
        matchRequantize(requant, acc, &relu, &innerBias);
    if (!scale)
      return failure();
    if (innerBias) {
      if (bias || !usableBias(innerBias))
        return failure();
      bias = innerBias;
    }
    if (!untouchedBetween(conv, requant, {input, filter}))
      return failure();

    auto outTy = llvm::cast<MemRefType>(requant.getOutputs()[0].getType());
    if (outTy.getRank() != 4 || outTy.getShape() != accTy.getShape())
      return failure();
    for (Operation *later = requant->getNextNode(); later;
         later = later->getNextNode())
      if (llvm::is_contained(later->getOperands(), acc) &&
          !llvm::isa<memref::DeallocOp>(later))
        return failure();

    if (!convBufferIsWalkable(input) ||
        !convBufferIsWalkable(requant.getOutputs()[0]))
      return failure();

    rewriter.setInsertionPoint(requant);
    Value chwFilter = relayoutConstantFilter(rewriter, conv, filter);
    if (!chwFilter)
      return failure();

    rewriter.create<DepthwiseConv2DInt8Op>(
        conv.getLoc(), input, chwFilter, bias, requant.getOutputs()[0],
        rewriter.getI64IntegerAttr(*stride), rewriter.getI64IntegerAttr(0),
        rewriter.getF32FloatAttr(scale->convertToFloat()),
        ActAttr::get(rewriter.getContext(), relu ? Act::RELU : Act::NONE),
        rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(0),
        DataflowAttr::get(rewriter.getContext(), Dataflow::WS));

    rewriter.eraseOp(requant);
    if (biasOp)
      rewriter.eraseOp(biasOp);
    rewriter.eraseOp(conv);
    eraseDeadZeroFill(rewriter, accZero);
    return success();
  }
};


// Folds a max-pool sitting on a convolution into the conv's own pooling.
//
// `tiled_conv_auto` pools the requantized i8 outputs with a plain max over the
// window -- `if (!initialized || opixel > running_max)` in conv_cpu -- which is
// what linalg.pooling_nhwc_max computes. Its out-of-bounds branch treats the
// padding as zero rather than -inf, so only `pool_padding = 0` is emitted;
// linalg has no pooling padding either (it is a separate pad), so nothing is
// lost.
/// `tiled_conv_auto` pools in the mvout pipeline, so a max-pool on the
/// convolution's result folds into the same call. It is **off by default**, and
/// not because the rewrite is wrong: the runtime's own CPU implementation of
/// the folded call gives exactly the unfolded answer. Folding removes one
/// intermediate buffer, and on the U280 board that changes the allocator's
/// behaviour enough that the convolution's output address starts alternating
/// between two bins -- 0x8c980, 0x8ca40, 0x8c980 -- which this hardware cannot
/// take (see docs/pipeline.md). Two hand-written `tiled_conv_auto` calls in a
/// row reproduce it with no compiler involved.
/// The copy of a convolution's output into the middle of a bigger buffer, as a
/// single padding amount. Only the shape `tiled_conv_auto` can express: the
/// same amount on every side of both spatial axes, and none on batch or
/// channel.
static bool zeroPaddedCopy(memref::SubViewOp sub, MemRefType paddedTy,
                           MemRefType innerTy, int64_t *padding) {
  SmallVector<int64_t> offsets, sizes, strides;
  for (OpFoldResult o : sub.getMixedOffsets()) {
    std::optional<int64_t> c = getConstantIntValue(o);
    if (!c)
      return false;
    offsets.push_back(*c);
  }
  for (OpFoldResult o : sub.getMixedSizes()) {
    std::optional<int64_t> c = getConstantIntValue(o);
    if (!c)
      return false;
    sizes.push_back(*c);
  }
  for (OpFoldResult o : sub.getMixedStrides()) {
    std::optional<int64_t> c = getConstantIntValue(o);
    if (!c || *c != 1)
      return false;
    strides.push_back(*c);
  }
  if (offsets.size() != 4 || sizes.size() != 4 || strides.size() != 4)
    return false;
  if (sizes != llvm::to_vector(innerTy.getShape()))
    return false;
  if (offsets[0] != 0 || offsets[3] != 0 ||
      paddedTy.getDimSize(0) != innerTy.getDimSize(0) ||
      paddedTy.getDimSize(3) != innerTy.getDimSize(3))
    return false;
  int64_t p = offsets[1];
  if (p < 1 || offsets[2] != p)
    return false;
  for (unsigned k = 1; k <= 2; k++)
    if (paddedTy.getDimSize(k) != innerTy.getDimSize(k) + 2 * p)
      return false;
  *padding = p;
  return true;
}

/// True when `v` names the same memory as `buffer`, possibly through views.
static bool aliasesBuffer(Value v, Value buffer) {
  for (unsigned step = 0; step < 8; step++) {
    if (v == buffer)
      return true;
    Operation *def = v.getDefiningOp();
    if (auto sub = llvm::dyn_cast_or_null<memref::SubViewOp>(def))
      v = sub.getSource();
    else if (auto collapse = llvm::dyn_cast_or_null<memref::CollapseShapeOp>(def))
      v = collapse.getSrc();
    else if (auto expand = llvm::dyn_cast_or_null<memref::ExpandShapeOp>(def))
      v = expand.getSrc();
    else
      return false;
  }
  return false;
}

/// The operation that wrote zero over the whole of `buffer` just before `use`,
/// if there is one. `--fill-to-memset` has not run yet, so it is still whatever
/// bufferization left: a `linalg.fill` or a `linalg.map` with no inputs.
static Operation *fillOfWholeBuffer(Operation *use, Value buffer) {
  for (Operation *prev = use->getPrevNode(); prev; prev = prev->getPrevNode()) {
    Value written, value;
    if (auto fill = llvm::dyn_cast<linalg::FillOp>(prev)) {
      if (fill.getOutputs().size() != 1 || fill.getInputs().size() != 1)
        return nullptr;
      written = fill.getOutputs()[0];
      value = fill.getInputs()[0];
    } else if (auto map = llvm::dyn_cast<linalg::MapOp>(prev)) {
      if (!map.getInputs().empty())
        return nullptr;
      auto yield = llvm::dyn_cast<linalg::YieldOp>(
          map.getRegion().front().getTerminator());
      if (!yield || yield.getNumOperands() != 1)
        return nullptr;
      written = map.getInit();
      value = yield.getOperand(0);
    } else {
      // Anything else may pass, so long as it does not write the buffer or a
      // view of it. A `memref.subview` between the fill and the copy is the
      // normal shape -- it is what the copy writes through -- and naming a
      // buffer is not writing it, which `writtenOperandsOf` has no answer for.
      if (llvm::isa<memref::SubViewOp, memref::CollapseShapeOp,
                    memref::ExpandShapeOp, memref::CastOp, memref::AllocOp,
                    memref::ViewOp>(prev))
        continue;
      std::optional<SmallVector<Value>> writes = writtenOperandsOf(prev);
      if (!writes)
        return nullptr;
      for (Value w : *writes)
        if (aliasesBuffer(w, buffer))
          return nullptr;
      continue;
    }
    if (written != buffer)
      continue;
    llvm::APInt i;
    if (matchPattern(value, m_ConstantInt(&i)) && i.isZero())
      return prev;
    return nullptr;
  }
  return nullptr;
}

class FoldMaxPoolIntoConv : public OpRewritePattern<Conv2DInt8Op> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DInt8Op conv,
                                PatternRewriter &rewriter) const final {
    if (conv.getPoolStride() != 0)
      return failure(); // already pooling
    Value convOut = conv.getOutput();
    if (!convOut.getDefiningOp<memref::AllocOp>())
      return failure();

    // A padded pool reaches this point as a **zero**-filled buffer with the
    // convolution's output copied into the middle of it -- which is exactly
    // what `pool_padding` computes, since `sp_tiled_conv`'s out-of-bounds
    // branch reads zero. Getting it to zero is `ZeroPadAMaxPool`'s job, back
    // where the relu that makes -inf and zero agree is still visible; by here
    // there is nothing left to argue.
    Value pooledInput = convOut;
    int64_t poolPadding = 0;
    Operation *padFill = nullptr, *padCopy = nullptr;
    Operation *next = nextUseOf(conv, convOut);
    if (auto copy = llvm::dyn_cast_or_null<memref::CopyOp>(next)) {
      auto sub = copy.getTarget().getDefiningOp<memref::SubViewOp>();
      if (copy.getSource() != convOut || !sub)
        return failure();
      Value padded = sub.getSource();
      auto paddedTy = llvm::dyn_cast<MemRefType>(padded.getType());
      auto innerTy = llvm::dyn_cast<MemRefType>(convOut.getType());
      if (!padded.getDefiningOp<memref::AllocOp>() || !paddedTy || !innerTy ||
          !paddedTy.hasStaticShape() || !innerTy.hasStaticShape() ||
          paddedTy.getRank() != 4)
        return failure();
      if (!zeroPaddedCopy(sub, paddedTy, innerTy, &poolPadding))
        return failure();
      padFill = fillOfWholeBuffer(copy, padded);
      if (!padFill)
        return failure();
      pooledInput = padded;
      padCopy = copy;
      next = nextUseOfBuffer(copy, padded);
    }

    // Bufferization puts the pool's own allocation and its identity fill
    // between the two, so "the next operation" is not the one that matters.
    auto pool = llvm::dyn_cast_or_null<linalg::PoolingNhwcMaxOp>(next);
    if (!pool || pool.getInputs().size() != 2 || pool.getOutputs().size() != 1)
      return failure();
    if (pool.getInputs()[0] != pooledInput)
      return failure();

    auto windowTy = llvm::dyn_cast<MemRefType>(pool.getInputs()[1].getType());
    if (!windowTy || windowTy.getRank() != 2 || !windowTy.hasStaticShape() ||
        windowTy.getShape()[0] != windowTy.getShape()[1])
      return failure();
    int64_t poolSize = windowTy.getShape()[0];

    auto pair = [](DenseIntElementsAttr a) -> std::optional<int64_t> {
      if (!a || a.getNumElements() != 2)
        return std::nullopt;
      auto it = a.value_begin<APInt>();
      int64_t x = (*it).getSExtValue(), y = (*(it + 1)).getSExtValue();
      return x == y ? std::optional<int64_t>(x) : std::nullopt;
    };
    std::optional<int64_t> poolStride = pair(pool.getStrides());
    std::optional<int64_t> poolDilation = pair(pool.getDilations());
    if (!poolStride || *poolStride < 1 || !poolDilation || *poolDilation != 1)
      return failure();

    auto pooledTy = llvm::dyn_cast<MemRefType>(pool.getOutputs()[0].getType());
    if (!pooledTy || pooledTy.getRank() != 4 || !pooledTy.hasStaticShape() ||
        !pooledTy.getElementType().isInteger(8))
      return failure();

    for (Operation *later = pool->getNextNode(); later;
         later = later->getNextNode())
      if ((llvm::is_contained(later->getOperands(), convOut) ||
           llvm::is_contained(later->getOperands(), pooledInput)) &&
          !llvm::isa<memref::DeallocOp>(later))
        return failure();

    Value pooled = pool.getOutputs()[0];
    // The call now writes the pooled buffer, which bufferization may have
    // allocated *after* it, so it takes the pool's place exactly -- and nothing
    // in between may have written what it reads.
    SmallVector<Value> reads = {conv.getInput(), conv.getFilter()};
    if (conv.getBias())
      reads.push_back(conv.getBias());
    if (!untouchedBetween(conv, pool, reads))
      return failure();
    // A max-pool starts from the reduction's identity. The runtime writes every
    // pooled output itself, so once it does the pooling that fill is dead.
    linalg::FillOp identity;
    for (Operation *prev = pool->getPrevNode(); prev; prev = prev->getPrevNode()) {
      if (auto fill = llvm::dyn_cast<linalg::FillOp>(prev))
        if (fill.getOutputs().size() == 1 && fill.getOutputs()[0] == pooled) {
          identity = fill;
          break;
        }
      if (llvm::is_contained(prev->getOperands(), pooled))
        break;
    }

    // No output window may be entirely padding: the runtime's pooled value
    // would then be a zero it invented rather than a real one, where the
    // pool's own reduction has nothing to reduce. `ZeroPadAMaxPool` checks the
    // same thing on tensors; it is cheap and it is the whole precondition.
    if (poolPadding) {
      auto innerTy = llvm::cast<MemRefType>(convOut.getType());
      for (unsigned k = 1; k <= 2; k++) {
        int64_t last = (pooledTy.getDimSize(k) - 1) * *poolStride;
        if (poolSize - 1 < poolPadding ||
            last > poolPadding + innerTy.getDimSize(k) - 1)
          return failure();
      }
    }

    rewriter.moveOpBefore(conv, pool);
    rewriter.eraseOp(pool);
    rewriter.modifyOpInPlace(conv, [&] {
      conv.getOutputMutable().assign(pooled);
      conv.setPoolSize(poolSize);
      conv.setPoolStride(*poolStride);
      conv.setPoolPadding(poolPadding);
    });
    if (identity)
      rewriter.eraseOp(identity);
    if (padCopy)
      rewriter.eraseOp(padCopy);
    if (padFill)
      rewriter.eraseOp(padFill);
    return success();
  }
};

// Folds an in-place relu into a gemmlir op that can apply it on the way out.
//
// Sound because the accelerator applies the activation before saturating, and
// relu and saturation to [-128, 127] commute: min(max(x,0),127) either way.
// Runs after the linalg conversion, on ops that carry an `act` attribute --
// gemmlir.matmul_i8 is not one of them, since reading the raw accumulators
// bypasses the pipeline the activation lives in.
template <typename OpTy>
class FoldReluIntoAct : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const final {
    if (op.getAct() != Act::NONE)
      return failure();
    Operation *next = op->getNextNode();
    if (!next || !isReluInPlace(next, op.getOutMat()))
      return failure();

    rewriter.eraseOp(next);
    rewriter.modifyOpInPlace(op, [&] {
      op.setActAttr(ActAttr::get(rewriter.getContext(), Act::RELU));
    });
    return success();
  }
};

/// Splits a shape into two groups of dimensions, preferring a trailing group
/// whose product is a multiple of the systolic array's side.
static SmallVector<ReassociationIndices> asMatrix(ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  unsigned split = rank - 1;
  int64_t trailing = 1;
  for (unsigned k = rank; k-- > 1;) {
    trailing *= shape[k];
    if (trailing % 16 == 0) {
      split = k;
      break;
    }
  }
  ReassociationIndices rows, cols;
  for (unsigned i = 0; i < split; i++)
    rows.push_back(i);
  for (unsigned i = split; i < rank; i++)
    cols.push_back(i);
  return {rows, cols};
}

/// A residual add as a quantized network writes it: two i8 tensors, each with
/// its own scale, summed, optionally relu'd, and requantized to i8 --
/// `trunci(clamp(fptosi(roundeven(relu?(a*sa + b*sb) / so))))`. That is
/// `tiled_resadd_auto`, and `--split-residual-add` is what leaves this shape
/// behind after taking the convolution's own requantization out of it.
///
/// The scales come back already divided by the output's, because the runtime
/// applies `A_scale` and `B_scale` on the way in and `C_scale` on the way out,
/// and folding the division into the two inputs leaves `C_scale` at 1. That
/// matters: `MVIN_SCALE` rounds *and clips to i8*, so a scale above 1 would
/// saturate an operand before the sum -- with the division folded in, each is
/// the ratio of an input's scale to the output's, at most 1 for a sum of two
/// values whose range the output covers.
static bool isScaledI8Add(linalg::GenericOp generic, double *lhs, double *rhs,
                          bool *relu) {
  if (generic.getInputs().size() != 2 || generic.getOutputs().size() != 1)
    return false;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;

  auto outTy = llvm::dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
  if (!outTy || !outTy.hasStaticShape() || !outTy.getElementType().isInteger(8) ||
      !outTy.getLayout().isIdentity() || outTy.getRank() < 2)
    return false;

  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 3 || !maps.back().isIdentity())
    return false;
  for (unsigned i = 0; i < 2; i++) {
    auto inTy = llvm::dyn_cast<MemRefType>(generic.getInputs()[i].getType());
    if (!inTy || inTy.getShape() != outTy.getShape() ||
        !inTy.getElementType().isInteger(8) || !inTy.getLayout().isIdentity() ||
        !isWholeShapeRead(maps[i], inTy.getShape()))
      return false;
  }

  Block &body = generic.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto trunc = yield.getOperand(0).getDefiningOp<arith::TruncIOp>();
  if (!trunc || !trunc.getType().isInteger(8))
    return false;
  int64_t lo = 0, hi = 0;
  std::optional<Value> clamped = matchClamp(trunc.getIn(), &lo, &hi);
  if (!clamped || hi != 127 || (lo != -128 && lo != 0))
    return false;
  auto toInt = clamped->getDefiningOp<arith::FPToSIOp>();
  if (!toInt)
    return false;
  auto round = toInt.getIn().getDefiningOp<math::RoundEvenOp>();
  if (!round)
    return false;

  auto constant = [](Value c, double *out) {
    llvm::APFloat f(0.0f);
    if (!matchPattern(c, m_ConstantFloat(&f)))
      return false;
    *out = f.convertToDouble();
    return true;
  };

  // The output scaling, then the activation, then the sum.
  double outScale = 1.0;
  *relu = lo == 0; // a clamp to [0, 127] is a relu the narrowing already carries
  Value v = round.getOperand();
  while (true) {
    double c = 0.0;
    if (auto div = v.getDefiningOp<arith::DivFOp>()) {
      if (!constant(div.getRhs(), &c) || !(c > 0.0))
        return false;
      outScale *= c;
      v = div.getLhs();
      continue;
    }
    if (auto mul = v.getDefiningOp<arith::MulFOp>()) {
      if (constant(mul.getRhs(), &c))
        v = mul.getLhs();
      else if (constant(mul.getLhs(), &c))
        v = mul.getRhs();
      else
        break;
      if (!(c > 0.0))
        return false;
      outScale /= c;
      continue;
    }
    if (std::optional<Value> inner = matchFloatRelu(v)) {
      *relu = true;
      v = *inner;
      continue;
    }
    break;
  }

  auto add = v.getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;
  // Each side: an i8 operand widened and multiplied by its own scale.
  auto side = [&](Value s, unsigned *arg, double *scale) {
    auto mul = s.getDefiningOp<arith::MulFOp>();
    if (!mul)
      return false;
    Value conv = mul.getLhs();
    if (!constant(mul.getRhs(), scale)) {
      conv = mul.getRhs();
      if (!constant(mul.getLhs(), scale))
        return false;
    }
    auto widen = conv.getDefiningOp<arith::SIToFPOp>();
    if (!widen || !(*scale > 0.0))
      return false;
    auto blockArg = llvm::dyn_cast<BlockArgument>(widen.getIn());
    if (!blockArg || blockArg.getArgNumber() > 1)
      return false;
    *arg = blockArg.getArgNumber();
    return true;
  };
  unsigned argA = 0, argB = 0;
  double scaleA = 0.0, scaleB = 0.0;
  if (!side(add.getLhs(), &argA, &scaleA) || !side(add.getRhs(), &argB, &scaleB))
    return false;
  if (argA == argB)
    return false;
  if (argA == 1)
    std::swap(scaleA, scaleB);
  *lhs = scaleA / outScale;
  *rhs = scaleB / outScale;
  return std::isfinite(*lhs) && std::isfinite(*rhs);
}

class LinalgScaledAddToGemmlir : public OpConversionPattern<linalg::GenericOp> {
public:
  LinalgScaledAddToGemmlir(MLIRContext *ctx, Dataflow dataflow)
      : OpConversionPattern(ctx), dataflow(dataflow) {}

  LogicalResult matchAndRewrite(linalg::GenericOp generic, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const final {
    double lhsScale = 0.0, rhsScale = 0.0;
    bool relu = false;
    if (!isScaledI8Add(generic, &lhsScale, &rhsScale, &relu))
      return rewriter.notifyMatchFailure(generic, "not a requantized scaled add");

    // The runtime takes two dimensions and a row stride, so the buffer is seen
    // as a matrix. Any split of a contiguous one is sound.
    auto outTy = llvm::cast<MemRefType>(generic.getOutputs()[0].getType());
    SmallVector<ReassociationIndices> groups = asMatrix(outTy.getShape());
    Location loc = generic.getLoc();
    auto flatten = [&](Value v) -> Value {
      if (outTy.getRank() == 2)
        return v;
      return rewriter.create<memref::CollapseShapeOp>(loc, v, groups);
    };

    rewriter.create<ResAddInt8Op>(
        loc, flatten(generic.getInputs()[0]), flatten(generic.getInputs()[1]),
        flatten(generic.getOutputs()[0]),
        rewriter.getF32FloatAttr(static_cast<float>(lhsScale)),
        rewriter.getF32FloatAttr(static_cast<float>(rhsScale)),
        rewriter.getF32FloatAttr(1.0f),
        ActAttr::get(rewriter.getContext(), relu ? Act::RELU : Act::NONE),
        DataflowAttr::get(rewriter.getContext(), dataflow));
    rewriter.eraseOp(generic);
    return success();
  }

private:
  Dataflow dataflow;
};

class ConvertLinalgToGemmlir
    : public impl::ConvertLinalgToGemmlirBase<ConvertLinalgToGemmlir> {
public:
  using impl::ConvertLinalgToGemmlirBase<
      ConvertLinalgToGemmlir>::ConvertLinalgToGemmlirBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                    gemmlir::GemmlirDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    std::optional<Dataflow> df = symbolizeDataflow(dataflow);
    if (!df) {
      module.emitError("unknown dataflow '") << dataflow << "', expected one of: os, ws, cpu";
      return signalPassFailure();
    }

    ConversionTarget target(getContext());
    target.addLegalDialect<gemmlir::GemmlirDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, func::FuncDialect,
                           arith::ArithDialect, scf::SCFDialect>();
    target.addIllegalOp<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::MatvecOp,
                        linalg::VecmatOp>();
    // Only a generic that spells out a saturating i8 add is offloaded; every
    // other one belongs to the ordinary lowering.
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp g) {
      bool relu = false;
      double lhs = 0.0, rhs = 0.0;
      return !isSaturatingI8Add(g, &relu) && !isScaledI8Add(g, &lhs, &rhs, &relu);
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgMatmulOpToGemmlir, LinalgBatchMatmulOpToGemmlir,
                 LinalgMatvecOpToGemmlir, LinalgVecmatOpToGemmlir,
                 LinalgSaturatingAddToGemmlir,
                 LinalgScaledAddToGemmlir>(&getContext(), *df);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();

    RewritePatternSet folds(&getContext());
    // Bias first: the requantize fold consumes the matmul, so the bias has to
    // be on it by then.
    // The conv has to exist before its pooling can be folded in, and the
    // greedy driver re-runs patterns until nothing changes, so ordering here is
    // a hint rather than a requirement.
    folds.add<FoldBiasIntoMatmul, FoldRequantizeIntoMatmul,
             FoldRequantizeIntoBatchMatmul, FuseQuantizedConv,
              FuseQuantizedDepthwiseConv,
              FoldConcatIntoConv, FoldConcatIntoElementwise,
              ElideRedundantBufferCopy,
              FoldPaddingIntoConv<Conv2DInt8Op>,
              FoldPaddingIntoConv<DepthwiseConv2DInt8Op>,
              FoldReluIntoAct<MatMulInt8ScaleOp>,
              FoldReluIntoAct<ResAddInt8Op>>(&getContext());
    if (fusePooling)
      folds.add<FoldMaxPoolIntoConv>(&getContext());
    // These folds walk a chain a step at a time -- a convolution takes its
    // requantization, then its padding, then its relu, then its slice of a
    // join -- so a deep model needs more scans than MLIR's default ten. RegNet
    // has 618 contractions and ran out, and the pass then failed with no
    // diagnostic at all.
    GreedyRewriteConfig config;
    config.setMaxIterations(64);
    if (failed(applyPatternsGreedily(module, std::move(folds), config)))
      signalPassFailure();
    if (failed(checkHostWrites(module)))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
