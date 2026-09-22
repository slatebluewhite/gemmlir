//===- PackInt8TableLookupPass.cpp --------------------------------*- C++ -*-===//
//
// Eight indices out of one load.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_PACKINT8TABLELOOKUP
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

constexpr uint64_t kHigh = 0x8080808080808080ull;
constexpr int64_t kLanes = 8;

/// The identity, allowing for the way a **batch of one** is written: bufferizing
/// a 4-D map on a unit axis turns `d0` into a constant `0`, so `isIdentity()`
/// says no to an ordinary elementwise loop
/// ([[mlir-unit-axis-is-a-constant-zero]]). Walk the map instead.
static bool isIdentityOverUnitAxes(AffineMap map, ArrayRef<int64_t> shape) {
  if ((int64_t)map.getNumResults() != (int64_t)shape.size() ||
      map.getNumDims() != map.getNumResults())
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

/// An i8 -> i8 table sweep is **72% of EfficientNet's elementwise work** and 29%
/// of a ViT's. `--table-for-i8-elementwise` leaves it as four instructions an
/// element -- `lb`, `add`, `lbu`, `sb` -- and it measures about **11 cycles**,
/// so something is waiting. The chain is two *dependent* loads: the byte, and
/// then the table entry that byte indexes.
///
/// One 8-byte load takes the first of them out of the chain. The bytes come out
/// of the register with a shift and a mask, so the body grows from four
/// instructions to about five and a quarter -- and measured on the board at
/// EfficientNet's shape, 16x16x576: **-44.5%**. Packing the eight *stores* back
/// into one word as well is worse (-37.7%), so the outputs stay bytes.
///
/// The index `--table-for-i8-elementwise` computes is `sext(q) + 128`, which for
/// the unsigned byte `u` in the word is `u ^ 0x80` -- so one `xor` of the whole
/// word does all eight, once.
class PackInt8TableLookup : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic->getNumResults() != 0 || generic.getInputs().size() != 1 ||
        generic.getOutputs().size() != 1)
      return failure();
    for (utils::IteratorType it : generic.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();
    Value src = generic.getInputs()[0], dst = generic.getOutputs()[0];
    if (src == dst)
      return failure();
    auto srcTy = llvm::dyn_cast<MemRefType>(src.getType());
    auto dstTy = llvm::dyn_cast<MemRefType>(dst.getType());
    if (!srcTy || !dstTy || !srcTy.getElementType().isInteger(8) ||
        !dstTy.getElementType().isInteger(8))
      return failure();
    if (!srcTy.hasStaticShape() || !dstTy.hasStaticShape() ||
        srcTy.getShape() != dstTy.getShape())
      return failure();
    // `memref.view` takes an identity `memref<?xi8>`, so both buffers have to
    // start where they say and be laid out densely.
    if (!srcTy.getLayout().isIdentity() || !dstTy.getLayout().isIdentity())
      return failure();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 ||
        !isIdentityOverUnitAxes(maps[0], srcTy.getShape()) ||
        !isIdentityOverUnitAxes(maps[1], dstTy.getShape()))
      return failure();
    int64_t total = srcTy.getNumElements();
    if (total <= 0 || total % kLanes != 0)
      return failure();

    // The body has to be exactly the lookup `--table-for-i8-elementwise` emits.
    Block &body = generic.getRegion().front();
    if (body.getNumArguments() != 2 || !body.getArgument(1).use_empty())
      return failure();
    auto yield = llvm::cast<linalg::YieldOp>(body.getTerminator());
    if (yield.getNumOperands() != 1)
      return failure();
    auto load = yield.getOperand(0).getDefiningOp<memref::LoadOp>();
    if (!load || load.getIndices().size() != 1)
      return failure();
    Value table = load.getMemRef();
    auto tableTy = llvm::dyn_cast<MemRefType>(table.getType());
    if (!tableTy || tableTy.getRank() != 1 || tableTy.getNumElements() != 256 ||
        !tableTy.getElementType().isInteger(8) ||
        table.getParentBlock() == &body)
      return failure();
    auto cast = load.getIndices()[0].getDefiningOp<arith::IndexCastOp>();
    if (!cast)
      return failure();
    auto add = cast.getIn().getDefiningOp<arith::AddIOp>();
    if (!add)
      return failure();
    IntegerAttr bias;
    Value widened;
    if (matchPattern(add.getRhs(), m_Constant(&bias)))
      widened = add.getLhs();
    else if (matchPattern(add.getLhs(), m_Constant(&bias)))
      widened = add.getRhs();
    else
      return failure();
    if (bias.getInt() != 128)
      return failure();
    auto ext = widened.getDefiningOp<arith::ExtSIOp>();
    if (!ext || ext.getIn() != body.getArgument(0))
      return failure();

    Location loc = generic.getLoc();
    Type i64 = rewriter.getI64Type();
    unsigned rank = srcTy.getRank();
    SmallVector<ReassociationIndices> group(1);
    for (unsigned i = 0; i < rank; i++)
      group[0].push_back(i);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value srcFlat = rank == 1 ? src
                              : rewriter.create<memref::CollapseShapeOp>(
                                    loc, src, group).getResult();
    Value dstFlat = rank == 1 ? dst
                              : rewriter.create<memref::CollapseShapeOp>(
                                    loc, dst, group).getResult();
    Value srcWords = rewriter.create<memref::ViewOp>(
        loc, MemRefType::get({total / kLanes}, i64), srcFlat, zero,
        ValueRange{});

    Value hi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(i64, (int64_t)kHigh));
    Value mask = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(i64, 0xFF));
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value words = rewriter.create<arith::ConstantIndexOp>(loc, total / kLanes);
    Value lanes = rewriter.create<arith::ConstantIndexOp>(loc, kLanes);

    auto loop = rewriter.create<scf::ForOp>(loc, c0, words, c1);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value w = loop.getInductionVar();
    Value raw = rewriter.create<memref::LoadOp>(loc, srcWords, w);
    // `sext(q) + 128` is `u ^ 0x80` on the unsigned byte, so one `xor` of the
    // whole word does all eight indices at once.
    Value flipped = rewriter.create<arith::XOrIOp>(loc, raw, hi);
    Value base = rewriter.create<arith::MulIOp>(loc, w, lanes);
    for (int64_t k = 0; k < kLanes; k++) {
      Value v = flipped;
      if (k) {
        Value sh = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(i64, 8 * k));
        v = rewriter.create<arith::ShRUIOp>(loc, flipped, sh);
      }
      Value byteVal = rewriter.create<arith::AndIOp>(loc, v, mask);
      Value idx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), byteVal);
      Value entry = rewriter.create<memref::LoadOp>(loc, table, idx);
      Value at = k ? rewriter.create<arith::AddIOp>(
                         loc, base,
                         rewriter.create<arith::ConstantIndexOp>(loc, k))
                         .getResult()
                   : base;
      rewriter.create<memref::StoreOp>(loc, entry, dstFlat, at);
    }

    rewriter.eraseOp(generic);
    return success();
  }
};

class PackInt8TableLookup_Pass
    : public impl::PackInt8TableLookupBase<PackInt8TableLookup_Pass> {
public:
  using impl::PackInt8TableLookupBase<
      PackInt8TableLookup_Pass>::PackInt8TableLookupBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, arith::ArithDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PackInt8TableLookup>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
