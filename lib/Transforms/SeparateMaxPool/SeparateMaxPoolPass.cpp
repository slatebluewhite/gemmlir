//===- SeparateMaxPoolPass.cpp -----------------------------------*- C++ -*-===//
//
// A two-dimensional maximum is two one-dimensional ones.
//
// `max` is associative and commutative, so a `kh x kw` window is `kh` rows of
// `kw` columns. Eight comparisons an output become four for a 3x3 -- and more
// than that, because the horizontal pass's result is shared between vertically
// adjacent outputs, so nine word loads an output become three and three.
//
// Measured at GoogLeNet's shapes on the same SWAR maximum: -65% to -68%, with
// identical outputs. For a maximum there is no rounding to argue about.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_SEPARATEMAXPOOL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The two numbers a pooling op carries as a dense vector attribute.
static bool pairOf(DenseIntElementsAttr attr, int64_t &a, int64_t &b) {
  if (!attr || attr.getNumElements() != 2)
    return false;
  auto it = attr.value_begin<APInt>();
  a = (*it).getSExtValue();
  ++it;
  b = (*it).getSExtValue();
  return true;
}

static DenseIntElementsAttr pair(OpBuilder &b, int64_t x, int64_t y) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get({2}, b.getI64Type()), ArrayRef<int64_t>{x, y});
}

class SeparateMaxPool : public impl::SeparateMaxPoolBase<SeparateMaxPool> {
public:
  using impl::SeparateMaxPoolBase<SeparateMaxPool>::SeparateMaxPoolBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    SmallVector<linalg::PoolingNhwcMaxOp> pools;
    getOperation().walk(
        [&](linalg::PoolingNhwcMaxOp p) { pools.push_back(p); });
    for (linalg::PoolingNhwcMaxOp p : pools)
      if (p->getBlock())
        (void)separate(p);
  }

private:
  LogicalResult separate(linalg::PoolingNhwcMaxOp pool);
};

LogicalResult SeparateMaxPool::separate(linalg::PoolingNhwcMaxOp pool) {
  auto inTy = llvm::dyn_cast<MemRefType>(pool.getInputs()[0].getType());
  auto winTy = llvm::dyn_cast<MemRefType>(pool.getInputs()[1].getType());
  auto outTy = llvm::dyn_cast<MemRefType>(pool.getOutputs()[0].getType());
  if (!inTy || !winTy || !outTy || inTy.getRank() != 4 || outTy.getRank() != 4 ||
      winTy.getRank() != 2 || !inTy.hasStaticShape() ||
      !outTy.hasStaticShape() || !winTy.hasStaticShape() ||
      !inTy.getLayout().isIdentity() || !outTy.getLayout().isIdentity())
    return failure();

  int64_t sy = 0, sx = 0, dy = 0, dx = 0;
  if (!pairOf(pool.getStrides(), sy, sx) ||
      !pairOf(pool.getDilations(), dy, dx) || sy < 1 || sx < 1 || dy < 1 ||
      dx < 1)
    return failure();
  // **Stride one only.** At stride two the row pass computes every row of the
  // padded input where the direct form visits only `outH * kh` of them, and
  // `outH * kh` is already the smaller number -- so separating is a loss on the
  // arithmetic alone, before any buffer is allocated. Measured: with the
  // stride-2 pools in, `squeezenet1_1` (whose pools are all stride 2) read
  // +14.1%.
  if (sy != 1 || sx != 1)
    return failure();
  const int64_t kh = winTy.getShape()[0], kw = winTy.getShape()[1];
  // A 2x2 saves one comparison and costs a whole buffer; 3x3 and up pays.
  if ((kh - 1) * (kw - 1) < 4)
    return failure();

  // The row pass keeps every input row and narrows the columns, so the vertical
  // pass reads exactly the rows the original did.
  const int64_t N = inTy.getShape()[0], padH = inTy.getShape()[1],
                C = inTy.getShape()[3];
  const int64_t outH = outTy.getShape()[1], outW = outTy.getShape()[2];

  if (outTy.getShape()[0] != N || outTy.getShape()[3] != C)
    return failure();
  if ((outH - 1) * sy + (kh - 1) * dy >= padH)
    return failure();
  // The row pass has to reach every column the vertical pass will read, which
  // is the horizontal half of the same question.
  const int64_t padW = inTy.getShape()[2];
  if ((outW - 1) * sx + (kw - 1) * dx >= padW)
    return failure();
  if (inTy.getElementType() != outTy.getElementType())
    return failure();

  // The value the output was initialised with is the identity for the maximum,
  // and the row buffer needs the same one.
  linalg::FillOp outFill;
  for (Operation *user : pool.getOutputs()[0].getUsers())
    if (auto f = llvm::dyn_cast<linalg::FillOp>(user)) {
      if (outFill)
        return failure();
      outFill = f;
    }
  if (!outFill || outFill.getInputs().size() != 1)
    return failure();
  Attribute neutral;
  if (!matchPattern(outFill.getInputs()[0], m_Constant(&neutral)))
    return failure();

  OpBuilder b(pool);
  Location loc = pool.getLoc();
  Type elem = inTy.getElementType();
  Value rows = b.create<memref::AllocOp>(
      loc, MemRefType::get({N, padH, outW, C}, elem), b.getI64IntegerAttr(64));
  Value neutralV = b.create<arith::ConstantOp>(
      loc, elem, llvm::cast<TypedAttr>(neutral));
  b.create<linalg::FillOp>(loc, ValueRange{neutralV}, ValueRange{rows});

  Value winH = b.create<memref::AllocOp>(
      loc, MemRefType::get({1, kw}, winTy.getElementType()),
      b.getI64IntegerAttr(64));
  Value winV = b.create<memref::AllocOp>(
      loc, MemRefType::get({kh, 1}, winTy.getElementType()),
      b.getI64IntegerAttr(64));

  b.create<linalg::PoolingNhwcMaxOp>(loc, TypeRange{},
                                     ValueRange{pool.getInputs()[0], winH},
                                     ValueRange{rows}, pair(b, 1, sx),
                                     pair(b, 1, dx));
  b.create<linalg::PoolingNhwcMaxOp>(loc, TypeRange{},
                                     ValueRange{rows, winV},
                                     ValueRange{pool.getOutputs()[0]},
                                     pair(b, sy, 1), pair(b, dy, 1));
  b.setInsertionPointAfter(pool);
  b.create<memref::DeallocOp>(loc, rows);
  b.create<memref::DeallocOp>(loc, winH);
  b.create<memref::DeallocOp>(loc, winV);

  pool.erase();
  return success();
}

} // namespace

} // namespace mlir::gemmlir
