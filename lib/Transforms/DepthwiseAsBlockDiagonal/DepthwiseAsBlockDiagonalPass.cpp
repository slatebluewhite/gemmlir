//===- DepthwiseAsBlockDiagonalPass.cpp --------------------------*- C++ -*-===//
//
// A depthwise convolution is a block-diagonal one, DIM channels at a time.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gemmlir/GemmlirOps.h"
#include "Gemmlir/GemmlirPasses.h"

namespace mlir::gemmlir {

#define GEN_PASS_DEF_DEPTHWISEASBLOCKDIAGONAL
#include "Gemmlir/GemmlirPasses.h.inc"

namespace {

/// The runtime runs a depthwise convolution **one channel a call**:
/// `tiled_conv_dw_auto` sets `pochs` to 1, so a layer costs `channels` calls of
/// whatever a call costs. EfficientNet has sixteen depthwise layers and some
/// eight thousand channels between them, and PC sampling puts 30% of the model
/// inside `gemmini_tiled_conv_dw_auto` -- 20.8% of the whole model on a single
/// `LOOP_CONV_WS` instruction, the core stalled issuing work the array finishes
/// immediately.
///
/// The array is 16x16 and a depthwise uses one column of it. Filling the other
/// fifteen with **zeros** costs nothing in the array -- it multiplies and adds
/// them either way -- and turns `DIM` calls into one. What it costs is weight
/// bytes: a `DIM x DIM` block-diagonal filter where a `DIM`-long one would do.
///
/// Measured on the board before this pass was written, a 240-channel 3x3
/// depthwise over 8x8:
///
/// | | ms | calls |
/// |---|---|---|
/// | one call a channel | 2.885 | 240 |
/// | block diagonal, 16 a call | **0.443** | 15 |
///
/// and the two outputs agreed on every one of the 15,360 bytes. It is the same
/// convolution: off the diagonal the weight is zero, so no channel can reach
/// another.
///
/// On the model set, byte for byte against both references:
///
/// | | ms | |
/// |---|---|---|
/// | `mnasnet0_5` | 72.12 -> **54.28** | -24.7% |
/// | `mobilenet_v2` | 90.32 -> **77.85** | -13.8% |
/// | `mobilenet_v3_small` | 134.66 -> **123.19** | -8.5% |
/// | `efficientnet_b0` | 380.84 -> **351.42** | -7.7% |
/// | `shufflenet_v2_x0_5` | 46.52 -> **45.79** | -1.6% |
/// | the set | 2459.65 -> **2387.20** | -2.9% |
///
/// EfficientNet gains least of the four because most of its depthwise time is
/// the array actually working -- 20.8% of the model sits on one `LOOP_CONV_WS`
/// -- and that does not go away. What goes away is the per-call cost, which is
/// what MnasNet and MobileNetV2, whose depthwise layers are smaller and more
/// numerous, were mostly paying.
///
/// **The remainder is worth taking too.** Refusing a channel count that is not
/// a multiple of `lanes` left ShuffleNet five layers, MnasNet three and
/// MobileNetV3 three -- 24, 72, 88 and 120 channels, a remainder of eight every
/// time. A second measurement with the partial group handled:
/// `shufflenet_v2_x0_5` -2.9%, `mobilenet_v3_small` -1.5%, `mnasnet0_5` -1.1%,
/// the set -0.2%, again byte for byte. Only a layer with fewer channels than
/// one group is left alone, and there is nothing to gather there.
class DepthwiseAsBlockDiagonal
    : public OpRewritePattern<DepthwiseConv2DInt8Op> {
public:
  DepthwiseAsBlockDiagonal(MLIRContext *ctx, int64_t lanes)
      : OpRewritePattern(ctx), lanes(lanes) {}

  LogicalResult matchAndRewrite(DepthwiseConv2DInt8Op dw,
                                PatternRewriter &rewriter) const final {
    auto inTy = llvm::dyn_cast<MemRefType>(dw.getInput().getType());
    auto outTy = llvm::dyn_cast<MemRefType>(dw.getOutput().getType());
    auto filTy = llvm::dyn_cast<MemRefType>(dw.getFilter().getType());
    if (!inTy || !outTy || !filTy || !inTy.hasStaticShape() ||
        !outTy.hasStaticShape() || !filTy.hasStaticShape())
      return failure();
    int64_t channels = filTy.getDimSize(0), kh = filTy.getDimSize(1),
            kw = filTy.getDimSize(2);
    if (channels <= lanes || inTy.getDimSize(3) != channels ||
        outTy.getDimSize(3) != channels)
      return failure();
    // The channel axis has to be the packed one on both sides, because the
    // groups are read and written as a slice of it.
    auto unitChannel = [](MemRefType t) {
      int64_t off = 0;
      SmallVector<int64_t> s;
      return succeeded(t.getStridesAndOffset(s, off)) && s.size() == 4 &&
             s[3] == 1;
    };
    if (!unitChannel(inTy) || !unitChannel(outTy))
      return failure();

    // The filter has to be a constant this pass can read and expand.
    auto get = dw.getFilter().getDefiningOp<memref::GetGlobalOp>();
    if (!get)
      return failure();
    auto module = dw->getParentOfType<ModuleOp>();
    auto global = module.lookupSymbol<memref::GlobalOp>(get.getNameAttr());
    if (!global || !global.getConstant() || !global.getInitialValue())
      return failure();
    auto dense =
        llvm::dyn_cast<DenseElementsAttr>(*global.getInitialValue());
    if (!dense || dense.getType().getNumElements() != channels * kh * kw)
      return failure();
    SmallVector<int8_t> flat;
    for (APInt v : dense.getValues<APInt>())
      flat.push_back((int8_t)v.getSExtValue());

    // groups x kh x kw x lanes x lanes, diagonal inside each group. A channel
    // count that is not a multiple of `lanes` leaves a remainder, and that last
    // group needs its **own** global: the lowering passes the filter's
    // out-channel count as the weight stride, so a narrower group cannot be a
    // slice of a wider one. MobileNetV3, MnasNet and ShuffleNet are all
    // 24, 72, 88 or 120 channels -- a remainder of eight every time.
    int64_t whole = channels / lanes, rest = channels % lanes;
    SmallVector<int8_t> block((size_t)whole * kh * kw * lanes * lanes, 0);
    SmallVector<int8_t> tail((size_t)rest * rest * kh * kw, 0);
    for (int64_t c = 0; c < channels; c++) {
      int64_t g = c / lanes, j = c % lanes;
      for (int64_t r = 0; r < kh; r++)
        for (int64_t s = 0; s < kw; s++) {
          int8_t v = flat[(c * kh + r) * kw + s];
          if (g < whole)
            block[(((g * kh + r) * kw + s) * lanes + j) * lanes + j] = v;
          else
            tail[((r * kw + s) * rest + j) * rest + j] = v;
        }
    }

    Location loc = dw.getLoc();
    auto i8 = rewriter.getI8Type();
    auto makeGlobal = [&](StringRef suffix, ArrayRef<int64_t> shape,
                          ArrayRef<int8_t> data) -> Value {
      auto ty = MemRefType::get(shape, i8);
      std::string name = (global.getSymName() + suffix).str();
      if (!module.lookupSymbol<memref::GlobalOp>(name)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(global);
        rewriter.create<memref::GlobalOp>(
            global.getLoc(), rewriter.getStringAttr(name),
            global.getSymVisibilityAttr(), TypeAttr::get(ty),
            DenseElementsAttr::get(RankedTensorType::get(shape, i8), data),
            /*constant=*/rewriter.getUnitAttr(), global.getAlignmentAttr());
      }
      return rewriter.create<memref::GetGlobalOp>(loc, ty, name);
    };
    auto blockTy = MemRefType::get({whole, kh, kw, lanes, lanes}, i8);
    Value blockBuf;
    if (whole)
      blockBuf = makeGlobal("_blockdiag", {whole, kh, kw, lanes, lanes}, block);
    Value tailBuf;
    if (rest)
      tailBuf = makeGlobal("_blockdiag_tail", {kh, kw, rest, rest}, tail);

    auto idx = [&](int64_t v) { return rewriter.getIndexAttr(v); };
    for (int64_t g = 0; g < whole + (rest ? 1 : 0); g++) {
      int64_t width = g < whole ? lanes : rest;
      SmallVector<OpFoldResult> inOff{idx(0), idx(0), idx(0), idx(g * lanes)};
      SmallVector<OpFoldResult> inSize{
          idx(inTy.getDimSize(0)), idx(inTy.getDimSize(1)),
          idx(inTy.getDimSize(2)), idx(width)};
      SmallVector<OpFoldResult> ones4(4, idx(1));
      Value in = rewriter.create<memref::SubViewOp>(loc, dw.getInput(), inOff,
                                                    inSize, ones4);
      SmallVector<OpFoldResult> outOff{idx(0), idx(0), idx(0), idx(g * lanes)};
      SmallVector<OpFoldResult> outSize{
          idx(outTy.getDimSize(0)), idx(outTy.getDimSize(1)),
          idx(outTy.getDimSize(2)), idx(width)};
      Value out = rewriter.create<memref::SubViewOp>(loc, dw.getOutput(),
                                                     outOff, outSize, ones4);
      // Rank reduced to the (kh, kw, in, out) the convolution wants; the block
      // is dense, so its out-channel stride is `lanes`, which is what the
      // lowering passes as the weight stride.
      Value fil;
      if (g < whole) {
        SmallVector<OpFoldResult> fOff{idx(g), idx(0), idx(0), idx(0), idx(0)};
        SmallVector<OpFoldResult> fSize{idx(1), idx(kh), idx(kw), idx(lanes),
                                        idx(lanes)};
        SmallVector<OpFoldResult> ones5(5, idx(1));
        auto fTy = llvm::cast<MemRefType>(
            memref::SubViewOp::inferRankReducedResultType(
                {kh, kw, lanes, lanes}, blockTy, fOff, fSize, ones5));
        fil = rewriter.create<memref::SubViewOp>(loc, fTy, blockBuf, fOff,
                                                 fSize, ones5);
      } else {
        fil = tailBuf;
      }
      Value bias;
      if (dw.getBias()) {
        auto biasTy = llvm::cast<MemRefType>(dw.getBias().getType());
        bias = rewriter.create<memref::SubViewOp>(
            loc, dw.getBias(), SmallVector<OpFoldResult>{idx(g * lanes)},
            SmallVector<OpFoldResult>{idx(width)},
            SmallVector<OpFoldResult>{idx(1)});
        (void)biasTy;
      }
      rewriter.create<Conv2DInt8Op>(
          loc, in, fil, bias, out,
          rewriter.getI64IntegerAttr(dw.getStride()),
          rewriter.getI64IntegerAttr(dw.getPadding()),
          rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1),
          dw.getScaleAttr(), ActAttr::get(rewriter.getContext(), dw.getAct()),
          rewriter.getI64IntegerAttr(dw.getPoolSize()),
          rewriter.getI64IntegerAttr(dw.getPoolStride()),
          rewriter.getI64IntegerAttr(dw.getPoolPadding()),
          DataflowAttr::get(rewriter.getContext(), dw.getDataflow()));
    }
    rewriter.eraseOp(dw);
    return success();
  }

private:
  int64_t lanes;
};

class DepthwiseAsBlockDiagonalPass
    : public impl::DepthwiseAsBlockDiagonalBase<DepthwiseAsBlockDiagonalPass> {
public:
  using impl::DepthwiseAsBlockDiagonalBase<
      DepthwiseAsBlockDiagonalPass>::DepthwiseAsBlockDiagonalBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    func::FuncDialect, GemmlirDialect>();
  }

  void runOnOperation() final {
    if (lanes < 2)
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<DepthwiseAsBlockDiagonal>(&getContext(), lanes);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::gemmlir
