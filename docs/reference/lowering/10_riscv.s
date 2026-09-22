	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	forward                         # -- Begin function forward
	.p2align	1
	.type	forward,@function
forward:                                # @forward
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -80
	.cfi_def_cfa_offset 80
	sd	ra, 72(sp)                      # 8-byte Folded Spill
	sd	s0, 64(sp)                      # 8-byte Folded Spill
	sd	s1, 56(sp)                      # 8-byte Folded Spill
	sd	s2, 48(sp)                      # 8-byte Folded Spill
	sd	s3, 40(sp)                      # 8-byte Folded Spill
	sd	s4, 32(sp)                      # 8-byte Folded Spill
	sd	s5, 24(sp)                      # 8-byte Folded Spill
	sd	s6, 16(sp)                      # 8-byte Folded Spill
	sd	s7, 8(sp)                       # 8-byte Folded Spill
	sd	s8, 0(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	addi	s0, sp, 80
	.cfi_def_cfa s0, 0
	li	a7, 0
	addi	t0, a0, 16
	lui	s2, %hi(.L__gemmlir_arena_forward_0)
	addi	s2, s2, %lo(.L__gemmlir_arena_forward_0)
	li	t2, 7
	li	a3, 15
	lui	a0, 270520
	li	a4, 256
	addi	s3, s2, 2047
	addi	a0, a0, 1337
	addi	t1, s3, 33
	fmv.w.x	fa5, a0
	lui	a6, 2
	j	.LBB0_2
.LBB0_1:                                #   in Loop: Header=BB0_2 Depth=1
	addi	a7, a7, 1
	add	t0, t0, a6
	addi	t1, t1, 2047
	addi	t1, t1, 1
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
                                        #       Child Loop BB0_8 Depth 3
                                        #         Child Loop BB0_11 Depth 4
	bgtz	a7, .LBB0_28
# %bb.3:                                #   in Loop: Header=BB0_2 Depth=1
	li	t3, 0
	mv	t4, t1
	mv	t5, t0
	j	.LBB0_5
.LBB0_4:                                #   in Loop: Header=BB0_5 Depth=2
	addi	t3, t3, 1
	addi	t5, t5, 1024
	addi	t4, t4, 1
.LBB0_5:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_8 Depth 3
                                        #         Child Loop BB0_11 Depth 4
	blt	t2, t3, .LBB0_1
# %bb.6:                                #   in Loop: Header=BB0_5 Depth=2
	li	t6, 0
	mv	s4, t4
	mv	s5, t5
	j	.LBB0_8
.LBB0_7:                                #   in Loop: Header=BB0_8 Depth=3
	addi	t6, t6, 1
	addi	s5, s5, 64
	addi	s4, s4, 128
.LBB0_8:                                #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_11 Depth 4
	blt	a3, t6, .LBB0_4
# %bb.9:                                #   in Loop: Header=BB0_8 Depth=3
	li	a2, 0
	mv	a5, s4
	mv	a0, s5
	j	.LBB0_11
.LBB0_10:                               #   in Loop: Header=BB0_11 Depth=4
	sb	s1, 24(a5)
	addi	a2, a2, 8
	addi	a0, a0, 32
	addi	a5, a5, 64
.LBB0_11:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        #       Parent Loop BB0_8 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	blt	a3, a2, .LBB0_7
# %bb.12:                               #   in Loop: Header=BB0_11 Depth=4
	flw	fa4, -16(a0)
	fmul.s	fa4, fa4, fa5
	fcvt.w.s	s1, fa4, rne
	addiw	a1, s1, 128
	bltu	a1, a4, .LBB0_14
# %bb.13:                               #   in Loop: Header=BB0_11 Depth=4
	slti	a1, s1, -128
	neg	a1, a1
	xori	s1, a1, 127
.LBB0_14:                               #   in Loop: Header=BB0_11 Depth=4
	sb	s1, -32(a5)
	flw	fa4, -12(a0)
	fmul.s	fa4, fa4, fa5
	fcvt.w.s	s1, fa4, rne
	addiw	a1, s1, 128
	bltu	a1, a4, .LBB0_16
# %bb.15:                               #   in Loop: Header=BB0_11 Depth=4
	slti	a1, s1, -128
	neg	a1, a1
	xori	s1, a1, 127
.LBB0_16:                               #   in Loop: Header=BB0_11 Depth=4
	sb	s1, -24(a5)
	flw	fa4, -8(a0)
	fmul.s	fa4, fa4, fa5
	fcvt.w.s	s1, fa4, rne
	addiw	a1, s1, 128
	bltu	a1, a4, .LBB0_18
# %bb.17:                               #   in Loop: Header=BB0_11 Depth=4
	slti	a1, s1, -128
	neg	a1, a1
	xori	s1, a1, 127
.LBB0_18:                               #   in Loop: Header=BB0_11 Depth=4
	sb	s1, -16(a5)
	flw	fa4, -4(a0)
	fmul.s	fa4, fa4, fa5
	fcvt.w.s	s1, fa4, rne
	addiw	a1, s1, 128
	bltu	a1, a4, .LBB0_20
# %bb.19:                               #   in Loop: Header=BB0_11 Depth=4
	slti	a1, s1, -128
	neg	a1, a1
	xori	s1, a1, 127
.LBB0_20:                               #   in Loop: Header=BB0_11 Depth=4
	sb	s1, -8(a5)
	flw	fa4, 0(a0)
	fmul.s	fa4, fa4, fa5
	fcvt.w.s	s1, fa4, rne
	addiw	a1, s1, 128
	bltu	a1, a4, .LBB0_22
# %bb.21:                               #   in Loop: Header=BB0_11 Depth=4
	slti	a1, s1, -128
	neg	a1, a1
	xori	s1, a1, 127
.LBB0_22:                               #   in Loop: Header=BB0_11 Depth=4
	sb	s1, 0(a5)
	flw	fa4, 4(a0)
	fmul.s	fa4, fa4, fa5
	fcvt.w.s	s1, fa4, rne
	addiw	a1, s1, 128
	bltu	a1, a4, .LBB0_24
# %bb.23:                               #   in Loop: Header=BB0_11 Depth=4
	slti	a1, s1, -128
	neg	a1, a1
	xori	s1, a1, 127
.LBB0_24:                               #   in Loop: Header=BB0_11 Depth=4
	sb	s1, 8(a5)
	flw	fa4, 8(a0)
	fmul.s	fa4, fa4, fa5
	fcvt.w.s	s1, fa4, rne
	addiw	a1, s1, 128
	bltu	a1, a4, .LBB0_26
# %bb.25:                               #   in Loop: Header=BB0_11 Depth=4
	slti	a1, s1, -128
	neg	a1, a1
	xori	s1, a1, 127
.LBB0_26:                               #   in Loop: Header=BB0_11 Depth=4
	sb	s1, 16(a5)
	flw	fa4, 12(a0)
	fmul.s	fa4, fa4, fa5
	fcvt.w.s	s1, fa4, rne
	addiw	a1, s1, 128
	bltu	a1, a4, .LBB0_10
# %bb.27:                               #   in Loop: Header=BB0_11 Depth=4
	slti	a1, s1, -128
	neg	a1, a1
	xori	s1, a1, 127
	j	.LBB0_10
.LBB0_28:
	#APP
	.insn r 123, 3, 7, zero, zero, zero
	#NO_APP
	call	gemmlir_flush
	addi	sp, sp, -176
	li	t4, 1
	lui	a7, 6
	lui	t0, %hi(.L__constant_16xi32)
	addi	t0, t0, %lo(.L__constant_16xi32)
	lui	t1, %hi(.L__constant_3x3x8x16xi8)
	addi	t1, t1, %lo(.L__constant_3x3x8x16xi8)
	addi	s3, s3, 1
	sd	zero, 64(sp)
	sd	zero, 72(sp)
	sd	zero, 80(sp)
	sd	zero, 88(sp)
	li	t2, 16
	li	t3, 8
	li	a5, 3
	lui	a6, 240831
	li	a0, 1
	li	a1, 16
	li	a2, 16
	li	a3, 8
	li	a4, 16
	sd	t4, 0(sp)
	sd	t4, 8(sp)
	sd	t4, 16(sp)
	sd	a5, 24(sp)
	li	a5, 16
	addi	s1, a6, -1959
	fmv.w.x	fa0, s1
	li	a6, 16
	addi	s1, a7, -1472
	add	s1, s1, s2
	sd	s3, 96(sp)
	sd	t1, 104(sp)
	sd	t0, 112(sp)
	sd	s1, 120(sp)
	li	a7, 1
	sd	t4, 160(sp)
	sd	t4, 128(sp)
	sd	zero, 136(sp)
	sd	zero, 144(sp)
	sd	zero, 152(sp)
	sd	t3, 32(sp)
	sd	t2, 40(sp)
	sd	t2, 48(sp)
	sd	zero, 56(sp)
	call	tiled_conv_stride_auto
	addi	sp, sp, 176
	lui	s1, 7
	addi	a0, s1, -1472
	add	a0, a0, s2
	li	a2, 288
	li	a1, 0
	call	gemmlir_memset
	lui	a0, 8
	addi	a0, a0, -672
	add	a0, a0, s2
	li	a2, 288
	li	a1, 0
	call	gemmlir_memset
	li	a7, 0
	addi	a3, s1, -1181
	li	a1, 15
	lui	a0, 1
	add	t0, s2, a3
	addi	a6, a0, 1088
	j	.LBB0_30
.LBB0_29:                               #   in Loop: Header=BB0_30 Depth=1
	addi	a7, a7, 1
	add	t0, t0, a6
.LBB0_30:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_33 Depth 2
                                        #       Child Loop BB0_36 Depth 3
                                        #         Child Loop BB0_38 Depth 4
	bgtz	a7, .LBB0_39
# %bb.31:                               #   in Loop: Header=BB0_30 Depth=1
	li	a4, 0
	mv	a2, t0
	j	.LBB0_33
.LBB0_32:                               #   in Loop: Header=BB0_33 Depth=2
	addi	a4, a4, 1
	addi	a2, a2, 288
.LBB0_33:                               #   Parent Loop BB0_30 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_36 Depth 3
                                        #         Child Loop BB0_38 Depth 4
	blt	a1, a4, .LBB0_29
# %bb.34:                               #   in Loop: Header=BB0_33 Depth=2
	li	s1, 0
	mv	a5, a2
	j	.LBB0_36
.LBB0_35:                               #   in Loop: Header=BB0_36 Depth=3
	addi	s1, s1, 1
	addi	a5, a5, 16
.LBB0_36:                               #   Parent Loop BB0_30 Depth=1
                                        #     Parent Loop BB0_33 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_38 Depth 4
	bgtz	s1, .LBB0_32
# %bb.37:                               #   in Loop: Header=BB0_36 Depth=3
	li	a3, 0
	bltz	a1, .LBB0_35
.LBB0_38:                               #   Parent Loop BB0_30 Depth=1
                                        #     Parent Loop BB0_33 Depth=2
                                        #       Parent Loop BB0_36 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	add	a0, a5, a3
	sb	zero, -3(a0)
	sb	zero, -2(a0)
	sb	zero, -1(a0)
	sb	zero, 0(a0)
	sb	zero, 1(a0)
	sb	zero, 2(a0)
	sb	zero, 3(a0)
	sb	zero, 4(a0)
	addi	a3, a3, 8
	bge	a1, a3, .LBB0_38
	j	.LBB0_35
.LBB0_39:
	li	a7, 0
	lui	a0, 7
	li	a1, 15
	lui	a2, 1
	addi	a3, a0, -909
	add	t0, s2, a3
	addi	a6, a2, 1088
	j	.LBB0_41
.LBB0_40:                               #   in Loop: Header=BB0_41 Depth=1
	addi	a7, a7, 1
	add	t0, t0, a6
.LBB0_41:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_44 Depth 2
                                        #       Child Loop BB0_47 Depth 3
                                        #         Child Loop BB0_49 Depth 4
	bgtz	a7, .LBB0_50
# %bb.42:                               #   in Loop: Header=BB0_41 Depth=1
	li	a4, 0
	mv	a2, t0
	j	.LBB0_44
.LBB0_43:                               #   in Loop: Header=BB0_44 Depth=2
	addi	a4, a4, 1
	addi	a2, a2, 288
.LBB0_44:                               #   Parent Loop BB0_41 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_47 Depth 3
                                        #         Child Loop BB0_49 Depth 4
	blt	a1, a4, .LBB0_40
# %bb.45:                               #   in Loop: Header=BB0_44 Depth=2
	li	s1, 0
	mv	a5, a2
	j	.LBB0_47
.LBB0_46:                               #   in Loop: Header=BB0_47 Depth=3
	addi	s1, s1, 1
	addi	a5, a5, 16
.LBB0_47:                               #   Parent Loop BB0_41 Depth=1
                                        #     Parent Loop BB0_44 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_49 Depth 4
	bgtz	s1, .LBB0_43
# %bb.48:                               #   in Loop: Header=BB0_47 Depth=3
	li	a3, 0
	bltz	a1, .LBB0_46
.LBB0_49:                               #   Parent Loop BB0_41 Depth=1
                                        #     Parent Loop BB0_44 Depth=2
                                        #       Parent Loop BB0_47 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	add	a0, a5, a3
	sb	zero, -3(a0)
	sb	zero, -2(a0)
	sb	zero, -1(a0)
	sb	zero, 0(a0)
	sb	zero, 1(a0)
	sb	zero, 2(a0)
	sb	zero, 3(a0)
	sb	zero, 4(a0)
	addi	a3, a3, 8
	bge	a1, a3, .LBB0_49
	j	.LBB0_46
.LBB0_50:
	mv	s4, sp
	mv	a3, sp
	addi	a0, a3, -96
	mv	sp, a0
	li	a7, 1
	li	a2, 16
	li	a4, 256
	lui	a6, 1
	lui	s1, 6
	lui	a5, 228023
	sd	a4, -32(a3)
	sd	a2, -24(a3)
	sd	a7, -16(a3)
	sd	a2, -64(a3)
	sd	a2, -56(a3)
	sd	a2, -48(a3)
	sd	a6, -40(a3)
	addi	a4, s1, -1472
	slli	a5, a5, 2
	add	a4, a4, s2
	addi	a5, a5, -273
	sd	a5, -96(a3)
	sd	a4, -88(a3)
	sd	zero, -80(a3)
	sd	a7, -72(a3)
	mv	a3, sp
	addi	a4, a3, -96
	mv	sp, a4
	li	s1, 288
	addi	s3, a6, 1088
	li	a6, 304
	lui	a1, 7
	sd	s1, -32(a3)
	sd	a2, -24(a3)
	sd	a7, -16(a3)
	sd	a2, -64(a3)
	sd	a2, -56(a3)
	sd	a2, -48(a3)
	sd	s3, -40(a3)
	addi	s5, a1, -1472
	add	s5, s5, s2
	sd	a5, -96(a3)
	sd	s5, -88(a3)
	sd	a6, -80(a3)
	sd	a7, -72(a3)
	mv	a2, sp
	addi	a1, a2, -16
	mv	sp, a1
	li	a3, 4
	sd	a3, -16(a2)
	sd	a0, -8(a2)
	mv	a0, sp
	addi	a2, a0, -16
	mv	sp, a2
	sd	a3, -16(a0)
	sd	a4, -8(a0)
	li	a0, 1
	call	memrefCopy
	lui	a0, 12
	addi	a0, a0, -384
	add	a7, s2, a0
	mv	sp, s4
	li	t0, 0
	li	t4, 15
	li	s6, 2
	lui	a6, 9
	j	.LBB0_52
.LBB0_51:                               #   in Loop: Header=BB0_52 Depth=1
	addi	t0, t0, 1
	add	a7, a7, a6
	add	s5, s5, s3
.LBB0_52:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_55 Depth 2
                                        #       Child Loop BB0_58 Depth 3
                                        #         Child Loop BB0_60 Depth 4
	bgtz	t0, .LBB0_61
# %bb.53:                               #   in Loop: Header=BB0_52 Depth=1
	li	t1, 0
	mv	t2, s5
	mv	t3, a7
	j	.LBB0_55
.LBB0_54:                               #   in Loop: Header=BB0_55 Depth=2
	addi	t1, t1, 1
	addi	a0, t3, 2047
	addi	t3, a0, 257
	addi	t2, t2, 288
.LBB0_55:                               #   Parent Loop BB0_52 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_58 Depth 3
                                        #         Child Loop BB0_60 Depth 4
	blt	t4, t1, .LBB0_51
# %bb.56:                               #   in Loop: Header=BB0_55 Depth=2
	li	t5, 0
	mv	t6, t2
	mv	s4, t3
	j	.LBB0_58
.LBB0_57:                               #   in Loop: Header=BB0_58 Depth=3
	addi	t5, t5, 1
	addi	s4, s4, 144
	addi	t6, t6, 16
.LBB0_58:                               #   Parent Loop BB0_52 Depth=1
                                        #     Parent Loop BB0_55 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_60 Depth 4
	blt	t4, t5, .LBB0_54
# %bb.59:                               #   in Loop: Header=BB0_58 Depth=3
	li	s7, 0
	mv	s1, t6
	mv	a0, s4
	bltz	s6, .LBB0_57
.LBB0_60:                               #   Parent Loop BB0_52 Depth=1
                                        #     Parent Loop BB0_55 Depth=2
                                        #       Parent Loop BB0_58 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	ld	a4, 32(s1)
	ld	a5, 40(s1)
	ld	a2, 0(s1)
	ld	a3, 8(s1)
	ld	a1, 16(s1)
	ld	s8, 24(s1)
	addi	s7, s7, 1
	sd	a4, 32(a0)
	sd	a5, 40(a0)
	sd	a2, 0(a0)
	sd	a3, 8(a0)
	sd	a1, 16(a0)
	sd	s8, 24(a0)
	addi	a0, a0, 48
	addi	s1, s1, 288
	bge	s6, s7, .LBB0_60
	j	.LBB0_57
.LBB0_61:
	#APP
	.insn r 123, 3, 7, zero, zero, zero
	#NO_APP
	call	gemmlir_flush
	addi	sp, sp, -96
	li	a5, 1
	sd	zero, 32(sp)
	sd	zero, 40(sp)
	sd	zero, 48(sp)
	sd	zero, 56(sp)
	li	s1, 16
	lui	a6, 12
	lui	s3, 8
	lui	a4, %hi(.L__constant_144x16xi8)
	addi	a4, a4, %lo(.L__constant_144x16xi8)
	lui	t0, 260096
	li	a0, 256
	li	a1, 16
	li	a2, 144
	li	a7, 144
	sd	a5, 64(sp)
	sd	zero, 72(sp)
	sd	zero, 80(sp)
	sd	a5, 88(sp)
	addi	a3, a6, -384
	fmv.w.x	fa0, t0
	addi	a6, s3, -384
	add	a3, a3, s2
	add	a6, a6, s2
	sd	s1, 0(sp)
	sd	s1, 8(sp)
	sd	s1, 16(sp)
	sd	a5, 24(sp)
	li	a5, 0
	fmv.s	fa1, fa0
	fmv.s	fa2, fa0
	fmv.s	fa3, fa0
	call	tiled_matmul_auto
	addi	sp, sp, 96
	li	t1, 0
	lui	a0, 21
	addi	a7, s3, -368
	lui	a2, 6
	li	s7, 15
	lui	s3, %hi(.L__constant_16xf32+16)
	addi	s3, s3, %lo(.L__constant_16xf32+16)
	lui	a3, 228133
	addi	a3, a3, 394
	fmv.w.x	fa5, a3
	lui	a3, 248917
	fmv.w.x	fa4, zero
	addi	t2, a0, -304
	add	a7, a7, s2
	addi	t0, a2, -1465
	addi	a0, a3, 1234
	add	t2, t2, s2
	add	t0, t0, s2
	fmv.w.x	fa3, a0
	lui	a6, 4
	j	.LBB0_63
.LBB0_62:                               #   in Loop: Header=BB0_63 Depth=1
	addi	t1, t1, 1
	add	t2, t2, a6
.LBB0_63:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_66 Depth 2
                                        #       Child Loop BB0_69 Depth 3
                                        #         Child Loop BB0_71 Depth 4
	bgtz	t1, .LBB0_72
# %bb.64:                               #   in Loop: Header=BB0_63 Depth=1
	li	t3, 0
	mv	t4, t2
	mv	t5, t0
	mv	t6, a7
	j	.LBB0_66
.LBB0_65:                               #   in Loop: Header=BB0_66 Depth=2
	addi	t3, t3, 1
	addi	t6, t6, 1024
	addi	t5, t5, 256
	addi	t4, t4, 1024
.LBB0_66:                               #   Parent Loop BB0_63 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_69 Depth 3
                                        #         Child Loop BB0_71 Depth 4
	blt	s7, t3, .LBB0_62
# %bb.67:                               #   in Loop: Header=BB0_66 Depth=2
	li	s4, 0
	mv	s5, t4
	mv	s8, t5
	mv	s6, t6
	j	.LBB0_69
.LBB0_68:                               #   in Loop: Header=BB0_69 Depth=3
	addi	s4, s4, 1
	addi	s6, s6, 64
	addi	s8, s8, 16
	addi	s5, s5, 64
.LBB0_69:                               #   Parent Loop BB0_63 Depth=1
                                        #     Parent Loop BB0_66 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_71 Depth 4
	blt	s7, s4, .LBB0_65
# %bb.70:                               #   in Loop: Header=BB0_69 Depth=3
	li	s1, 0
	mv	a4, s5
	mv	a0, s6
	mv	a2, s3
	bltz	s7, .LBB0_68
.LBB0_71:                               #   Parent Loop BB0_63 Depth=1
                                        #     Parent Loop BB0_66 Depth=2
                                        #       Parent Loop BB0_69 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	add	a5, s8, s1
	lw	a1, -16(a0)
	lb	a3, -7(a5)
	flw	fa2, -16(a2)
	flw	fa1, -12(a2)
	flw	fa0, -8(a2)
	flw	ft0, -4(a2)
	fcvt.s.w	ft1, a1
	fmadd.s	fa2, ft1, fa5, fa2
	fcvt.s.w	ft1, a3
	fmadd.s	fa2, ft1, fa3, fa2
	fmax.s	fa2, fa2, fa4
	fsw	fa2, -16(a4)
	lb	a1, -6(a5)
	lw	a3, -12(a0)
	fcvt.s.w	fa2, a1
	fcvt.s.w	ft1, a3
	fmadd.s	fa1, ft1, fa5, fa1
	fmadd.s	fa2, fa2, fa3, fa1
	fmax.s	fa2, fa2, fa4
	fsw	fa2, -12(a4)
	lb	a1, -5(a5)
	lw	a3, -8(a0)
	fcvt.s.w	fa2, a1
	fcvt.s.w	fa1, a3
	fmadd.s	fa1, fa1, fa5, fa0
	fmadd.s	fa2, fa2, fa3, fa1
	fmax.s	fa2, fa2, fa4
	fsw	fa2, -8(a4)
	lb	a1, -4(a5)
	lw	a3, -4(a0)
	fcvt.s.w	fa2, a1
	fcvt.s.w	fa1, a3
	fmadd.s	fa1, fa1, fa5, ft0
	fmadd.s	fa2, fa2, fa3, fa1
	fmax.s	fa2, fa2, fa4
	fsw	fa2, -4(a4)
	lb	a1, -3(a5)
	lw	a3, 0(a0)
	flw	fa2, 0(a2)
	flw	fa1, 4(a2)
	flw	fa0, 8(a2)
	flw	ft0, 12(a2)
	fcvt.s.w	ft1, a3
	fmadd.s	fa2, ft1, fa5, fa2
	fcvt.s.w	ft1, a1
	fmadd.s	fa2, ft1, fa3, fa2
	fmax.s	fa2, fa2, fa4
	fsw	fa2, 0(a4)
	lw	a1, 4(a0)
	lb	a3, -2(a5)
	fcvt.s.w	fa2, a1
	fmadd.s	fa2, fa2, fa5, fa1
	fcvt.s.w	fa1, a3
	fmadd.s	fa2, fa1, fa3, fa2
	fmax.s	fa2, fa2, fa4
	fsw	fa2, 4(a4)
	lw	a1, 8(a0)
	lb	a3, -1(a5)
	addi	s1, s1, 8
	fcvt.s.w	fa2, a1
	fmadd.s	fa2, fa2, fa5, fa0
	fcvt.s.w	fa1, a3
	fmadd.s	fa2, fa1, fa3, fa2
	fmax.s	fa2, fa2, fa4
	fsw	fa2, 8(a4)
	lb	a1, 0(a5)
	lw	a3, 12(a0)
	addi	a2, a2, 32
	addi	a0, a0, 32
	fcvt.s.w	fa2, a1
	fcvt.s.w	fa1, a3
	fmadd.s	fa1, fa1, fa5, ft0
	fmadd.s	fa2, fa2, fa3, fa1
	fmax.s	fa2, fa2, fa4
	fsw	fa2, 12(a4)
	addi	a4, a4, 32
	bge	s7, s1, .LBB0_71
	j	.LBB0_68
.LBB0_72:
	li	a7, 0
	lui	a0, 25
	li	a1, 7
	li	a2, 15
	lui	a3, 1046528
	addi	t0, a0, -304
	add	t0, t0, s2
	lui	a6, 1
	j	.LBB0_74
.LBB0_73:                               #   in Loop: Header=BB0_74 Depth=1
	addi	a7, a7, 1
	add	t0, t0, a6
.LBB0_74:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_77 Depth 2
                                        #       Child Loop BB0_80 Depth 3
                                        #         Child Loop BB0_82 Depth 4
	bgtz	a7, .LBB0_83
# %bb.75:                               #   in Loop: Header=BB0_74 Depth=1
	li	t1, 0
	mv	t2, t0
	j	.LBB0_77
.LBB0_76:                               #   in Loop: Header=BB0_77 Depth=2
	addi	t1, t1, 1
	addi	t2, t2, 512
.LBB0_77:                               #   Parent Loop BB0_74 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_80 Depth 3
                                        #         Child Loop BB0_82 Depth 4
	blt	a1, t1, .LBB0_73
# %bb.78:                               #   in Loop: Header=BB0_77 Depth=2
	li	a0, 0
	mv	a5, t2
	j	.LBB0_80
.LBB0_79:                               #   in Loop: Header=BB0_80 Depth=3
	addi	a0, a0, 1
	addi	a5, a5, 64
.LBB0_80:                               #   Parent Loop BB0_74 Depth=1
                                        #     Parent Loop BB0_77 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_82 Depth 4
	blt	a1, a0, .LBB0_76
# %bb.81:                               #   in Loop: Header=BB0_80 Depth=3
	li	s1, 0
	mv	a4, a5
	bltz	a2, .LBB0_79
.LBB0_82:                               #   Parent Loop BB0_74 Depth=1
                                        #     Parent Loop BB0_77 Depth=2
                                        #       Parent Loop BB0_80 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	sw	a3, -16(a4)
	sw	a3, -12(a4)
	sw	a3, -8(a4)
	sw	a3, -4(a4)
	sw	a3, 0(a4)
	sw	a3, 4(a4)
	sw	a3, 8(a4)
	sw	a3, 12(a4)
	addi	s1, s1, 8
	addi	a4, a4, 32
	bge	a2, s1, .LBB0_82
	j	.LBB0_79
.LBB0_83:
	li	t0, 0
	lui	a0, 21
	lui	a1, 25
	li	t6, 7
	li	a2, 15
	lui	a6, 4
	addi	t1, a0, 704
	addi	t2, a1, -304
	add	t1, t1, s2
	add	t2, t2, s2
	lui	a7, 1
	j	.LBB0_85
.LBB0_84:                               #   in Loop: Header=BB0_85 Depth=1
	addi	t0, t0, 1
	add	t1, t1, a6
	add	t2, t2, a7
.LBB0_85:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_88 Depth 2
                                        #       Child Loop BB0_91 Depth 3
                                        #         Child Loop BB0_94 Depth 4
	bgtz	t0, .LBB0_223
# %bb.86:                               #   in Loop: Header=BB0_85 Depth=1
	li	t3, 0
	mv	t4, t2
	mv	t5, t1
	j	.LBB0_88
.LBB0_87:                               #   in Loop: Header=BB0_88 Depth=2
	addi	t3, t3, 1
	addi	t5, t5, 2047
	addi	t5, t5, 1
	addi	t4, t4, 512
.LBB0_88:                               #   Parent Loop BB0_85 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_91 Depth 3
                                        #         Child Loop BB0_94 Depth 4
	blt	t6, t3, .LBB0_84
# %bb.89:                               #   in Loop: Header=BB0_88 Depth=2
	li	s3, 0
	mv	a3, t4
	mv	a1, t5
	j	.LBB0_91
.LBB0_90:                               #   in Loop: Header=BB0_91 Depth=3
	addi	s3, s3, 1
	addi	a1, a1, 128
	addi	a3, a3, 64
.LBB0_91:                               #   Parent Loop BB0_85 Depth=1
                                        #     Parent Loop BB0_88 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_94 Depth 4
	blt	t6, s3, .LBB0_87
# %bb.92:                               #   in Loop: Header=BB0_91 Depth=3
	li	s1, 0
	mv	a5, a3
	mv	a4, a1
	j	.LBB0_94
.LBB0_93:                               #   in Loop: Header=BB0_94 Depth=4
	fmax.s	fa5, fa5, fa4
	addi	s1, s1, 8
	addi	a4, a4, 32
	fsw	fa5, 12(a5)
	addi	a5, a5, 32
.LBB0_94:                               #   Parent Loop BB0_85 Depth=1
                                        #     Parent Loop BB0_88 Depth=2
                                        #       Parent Loop BB0_91 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	blt	a2, s1, .LBB0_90
# %bb.95:                               #   in Loop: Header=BB0_94 Depth=4
	flw	fa5, -16(a5)
	flw	fa3, -1024(a4)
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_97
# %bb.96:                               #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_97:                               #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_99
# %bb.98:                               #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_99:                               #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, -960(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_101
# %bb.100:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_101:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_103
# %bb.102:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_103:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 0(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_105
# %bb.104:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_105:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_107
# %bb.106:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_107:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 64(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_109
# %bb.108:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_109:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_111
# %bb.110:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_111:                              #   in Loop: Header=BB0_94 Depth=4
	fmax.s	fa5, fa5, fa4
	fsw	fa5, -16(a5)
	flw	fa5, -12(a5)
	flw	fa3, -1020(a4)
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_113
# %bb.112:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_113:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_115
# %bb.114:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_115:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, -956(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_117
# %bb.116:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_117:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_119
# %bb.118:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_119:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 4(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_121
# %bb.120:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_121:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_123
# %bb.122:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_123:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 68(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_125
# %bb.124:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_125:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_127
# %bb.126:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_127:                              #   in Loop: Header=BB0_94 Depth=4
	fmax.s	fa5, fa5, fa4
	fsw	fa5, -12(a5)
	flw	fa5, -8(a5)
	flw	fa3, -1016(a4)
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_129
# %bb.128:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_129:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_131
# %bb.130:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_131:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, -952(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_133
# %bb.132:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_133:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_135
# %bb.134:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_135:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 8(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_137
# %bb.136:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_137:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_139
# %bb.138:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_139:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 72(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_141
# %bb.140:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_141:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_143
# %bb.142:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_143:                              #   in Loop: Header=BB0_94 Depth=4
	fmax.s	fa5, fa5, fa4
	fsw	fa5, -8(a5)
	flw	fa5, -4(a5)
	flw	fa3, -1012(a4)
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_145
# %bb.144:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_145:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_147
# %bb.146:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_147:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, -948(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_149
# %bb.148:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_149:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_151
# %bb.150:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_151:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 12(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_153
# %bb.152:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_153:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_155
# %bb.154:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_155:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 76(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_157
# %bb.156:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_157:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_159
# %bb.158:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_159:                              #   in Loop: Header=BB0_94 Depth=4
	fmax.s	fa5, fa5, fa4
	fsw	fa5, -4(a5)
	flw	fa5, 0(a5)
	flw	fa3, -1008(a4)
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_161
# %bb.160:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_161:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_163
# %bb.162:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_163:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, -944(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_165
# %bb.164:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_165:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_167
# %bb.166:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_167:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 16(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_169
# %bb.168:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_169:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_171
# %bb.170:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_171:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 80(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_173
# %bb.172:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_173:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_175
# %bb.174:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_175:                              #   in Loop: Header=BB0_94 Depth=4
	fmax.s	fa5, fa5, fa4
	fsw	fa5, 0(a5)
	flw	fa5, 4(a5)
	flw	fa3, -1004(a4)
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_177
# %bb.176:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_177:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_179
# %bb.178:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_179:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, -940(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_181
# %bb.180:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_181:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_183
# %bb.182:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_183:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 20(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_185
# %bb.184:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_185:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_187
# %bb.186:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_187:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 84(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_189
# %bb.188:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_189:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_191
# %bb.190:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_191:                              #   in Loop: Header=BB0_94 Depth=4
	fmax.s	fa5, fa5, fa4
	fsw	fa5, 4(a5)
	flw	fa5, 8(a5)
	flw	fa3, -1000(a4)
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_193
# %bb.192:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_193:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_195
# %bb.194:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_195:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, -936(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_197
# %bb.196:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_197:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_199
# %bb.198:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_199:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 24(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_201
# %bb.200:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_201:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_203
# %bb.202:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_203:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 88(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_205
# %bb.204:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_205:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_207
# %bb.206:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_207:                              #   in Loop: Header=BB0_94 Depth=4
	fmax.s	fa5, fa5, fa4
	fsw	fa5, 8(a5)
	flw	fa5, 12(a5)
	flw	fa3, -996(a4)
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_209
# %bb.208:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_209:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_211
# %bb.210:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_211:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, -932(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_213
# %bb.212:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_213:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_215
# %bb.214:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_215:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 28(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_217
# %bb.216:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_217:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_219
# %bb.218:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
.LBB0_219:                              #   in Loop: Header=BB0_94 Depth=4
	flw	fa3, 92(a4)
	fmax.s	fa5, fa5, fa4
	feq.s	a0, fa5, fa5
	fmv.s	fa4, fa3
	bnez	a0, .LBB0_221
# %bb.220:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa4, fa5
.LBB0_221:                              #   in Loop: Header=BB0_94 Depth=4
	feq.s	a0, fa3, fa3
	bnez	a0, .LBB0_93
# %bb.222:                              #   in Loop: Header=BB0_94 Depth=4
	fmv.s	fa5, fa3
	j	.LBB0_93
.LBB0_223:
	lui	a0, 1
	lui	s3, 1
	call	malloc
	li	a6, 0
	lui	a1, 25
	addi	a7, a0, 8
	li	a3, 7
	addi	a1, a1, -192
	add	s2, s2, a1
	li	t3, 15
	j	.LBB0_225
.LBB0_224:                              #   in Loop: Header=BB0_225 Depth=1
	addi	a6, a6, 1
	add	s2, s2, s3
	add	a7, a7, s3
.LBB0_225:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_228 Depth 2
                                        #       Child Loop BB0_231 Depth 3
                                        #         Child Loop BB0_233 Depth 4
	bgtz	a6, .LBB0_234
# %bb.226:                              #   in Loop: Header=BB0_225 Depth=1
	li	t0, 0
	mv	t1, a7
	mv	t2, s2
	j	.LBB0_228
.LBB0_227:                              #   in Loop: Header=BB0_228 Depth=2
	addi	t0, t0, 1
	addi	t2, t2, 512
	addi	t1, t1, 32
.LBB0_228:                              #   Parent Loop BB0_225 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_231 Depth 3
                                        #         Child Loop BB0_233 Depth 4
	blt	a3, t0, .LBB0_224
# %bb.229:                              #   in Loop: Header=BB0_228 Depth=2
	li	t4, 0
	mv	a4, t1
	mv	s1, t2
	j	.LBB0_231
.LBB0_230:                              #   in Loop: Header=BB0_231 Depth=3
	addi	t4, t4, 1
	addi	s1, s1, 4
	addi	a4, a4, 256
.LBB0_231:                              #   Parent Loop BB0_225 Depth=1
                                        #     Parent Loop BB0_228 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_233 Depth 4
	blt	t3, t4, .LBB0_227
# %bb.232:                              #   in Loop: Header=BB0_231 Depth=3
	li	a5, 0
	mv	a2, a4
	mv	a1, s1
	bltz	a3, .LBB0_230
.LBB0_233:                              #   Parent Loop BB0_225 Depth=1
                                        #     Parent Loop BB0_228 Depth=2
                                        #       Parent Loop BB0_231 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	flw	fa5, -128(a1)
	fsw	fa5, -8(a2)
	flw	fa5, -64(a1)
	fsw	fa5, -4(a2)
	flw	fa5, 0(a1)
	fsw	fa5, 0(a2)
	flw	fa5, 64(a1)
	addi	a5, a5, 4
	addi	a1, a1, 256
	fsw	fa5, 4(a2)
	addi	a2, a2, 16
	bge	a3, a5, .LBB0_233
	j	.LBB0_230
.LBB0_234:
	addi	sp, s0, -80
	.cfi_def_cfa sp, 80
	ld	ra, 72(sp)                      # 8-byte Folded Reload
	ld	s0, 64(sp)                      # 8-byte Folded Reload
	ld	s1, 56(sp)                      # 8-byte Folded Reload
	ld	s2, 48(sp)                      # 8-byte Folded Reload
	ld	s3, 40(sp)                      # 8-byte Folded Reload
	ld	s4, 32(sp)                      # 8-byte Folded Reload
	ld	s5, 24(sp)                      # 8-byte Folded Reload
	ld	s6, 16(sp)                      # 8-byte Folded Reload
	ld	s7, 8(sp)                       # 8-byte Folded Reload
	ld	s8, 0(sp)                       # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s1
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	.cfi_restore s6
	.cfi_restore s7
	.cfi_restore s8
	addi	sp, sp, 80
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	forward, .Lfunc_end0-forward
	.cfi_endproc
                                        # -- End function
	.type	.L__gemmlir_arena_forward_0,@object # @__gemmlir_arena_forward_0
	.local	.L__gemmlir_arena_forward_0
	.comm	.L__gemmlir_arena_forward_0,106176,64
	.type	.L__constant_16xf32,@object     # @__constant_16xf32
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.L__constant_16xf32:
	.word	0xbdbeed48                      # float -0.0932260155
	.word	0xbba4a3c1                      # float -0.0050244038
	.word	0xbc74cc5c                      # float -0.0149413012
	.word	0x3e958ab7                      # float 0.292073935
	.word	0x3c747d30                      # float 0.0149224252
	.word	0x3e7dbb37                      # float 0.24778448
	.word	0x3e7f85ee                      # float 0.249534339
	.word	0xbdcdfcd3                      # float -0.10057988
	.word	0xbd2c117a                      # float -0.0420088544
	.word	0x3eeb4959                      # float 0.459543973
	.word	0xbe9c3b30                      # float -0.305139065
	.word	0xbe7b76d6                      # float -0.245570511
	.word	0x3edab5ee                      # float 0.427169263
	.word	0xbe702ab6                      # float -0.234537929
	.word	0xbe25379e                      # float -0.161344975
	.word	0xbe2b258a                      # float -0.167135388
	.size	.L__constant_16xf32, 64

	.type	.L__constant_3x3x8x16xi8,@object # @__constant_3x3x8x16xi8
	.p2align	6, 0x0
.L__constant_3x3x8x16xi8:
	.ascii	"\377)\252\261\354\264I?\347\257DvJU\342/"
	.ascii	"\032]42\275\356\254\344\332^\324\313\004\354\367\264"
	.ascii	"\276&\252>\017QG\032\"\362\031\306\311L\272\377"
	.ascii	"\246\304\331\000<F\347IHO\3248C\203\026\231"
	.ascii	"\307\372\nT>P\275\331?.\334\035?\177:\017"
	.ascii	"\316\0054:9\252(IHa\305\316\327\203X\357"
	.ascii	"A\346\307\320I\2605/0\24446!\0133\340"
	.ascii	"\035\365Q\035F6\367\363!\317\005\224\305\350\364\302"
	.ascii	"4W\323\313\374\221\006\330\272\347\302!\340\225\337\247"
	.ascii	"\343O4\270%\267\357\023\017\317\360c\376\207\272\324"
	.ascii	"\3265\255\300&W.\312!O\304\325GA1\364"
	.ascii	"\303bV4\321\331\253\310\337\361\367{\024\201\243%"
	.ascii	"T\020\354\342Q\363\345\021\337F?\005JS\022\251"
	.ascii	"=\017L6\302l\317\271\022\t\356\326'\034\225\262"
	.ascii	"\307\315\376\3202\3320\004\024\027\312\\\nct\305"
	.ascii	"5\003#\003\3608\270:\323\247\322\0029\365<;"
	.ascii	"\260[AR\312\245\307\030\0378\363\323\372\\\025,"
	.ascii	"\355\230\244\343\342N:\032\035%C\251\013\210#O"
	.ascii	"#K\361.#dV\324\276\340\275\",\021\327\365"
	.ascii	"\347F#\236\0016\336\303\335\374\346\3325\020\343\002"
	.ascii	"+\325D\3045\277&\357\033\360\335OE\270T\226"
	.ascii	"9\031AN#V)\270\266\240I \033\273\365\254"
	.ascii	"\022\333\004\240\312\341\003\267\030\277T\006%\030\315\365"
	.ascii	"\364\367\f\274#\231\321&\013\242\335\307\030\235\263\214"
	.ascii	"\270\\\377\256\036Y\341\037J\326\316\315\337\210\024\362"
	.ascii	"\243\024<\336\325\253\256\334<\346\355\000:1E\317"
	.ascii	"Q'\001\270\365\343,\332FR\005\252)\272T\350"
	.ascii	"\332\322\252I%\r\023\307(F\275C\340`\233\307"
	.ascii	"/>\033\332\fWU\334B\253\306\3473\0133\321"
	.ascii	"\325)\275%D\327\303J\352H\300\352\f\022d0"
	.ascii	"\264\252>\026P\352\345\324\007\365\316\030\264\327\262\001"
	.ascii	"\004\025\377\256\020\354L?\030d\347RK\377w\035"
	.ascii	"\332\025\320\371\034=\023\352\352V\320\261\366\034=1"
	.ascii	"\277\356\265\361\020\311H\007;\352\bpJ\312\264K"
	.ascii	"\354\231\325\325\2672\307EH\235C\024**``"
	.ascii	"T\346=\254\007\345\274\000\335\344\000\334#\263J\213"
	.ascii	"\005\300.\233*H2\n&3\r\f5A\213\354"
	.ascii	"\374\006\004\336Y?\325\321\314\264\347Q\022\255/\243"
	.ascii	"\274\352\270\307+\013\355\347\034\rT\005\272V\362]"
	.ascii	"\027B\355Q&\254O+\005\346\256\214\265\226\207\242"
	.ascii	"\032\245\267d\376j\276\031\301\272\024\252\020\255\254V"
	.ascii	"\330\357\354\001\322\223\336+\365K\305{\323\263\272\320"
	.ascii	"I\274$\036\362dH\354D\361\3506\304\006\022Z"
	.ascii	"\301\234\365\001\007\300\033\355\333Z?\356\275\034\263\321"
	.ascii	"\316_\".\324\b\366=\013\0372\322\366pM\222"
	.ascii	">\315\275\344N\300\271\016B\030\034\007B\240f\244"
	.ascii	"\316\026\276\2451h\013&\356W\257V.\007\216\207"
	.ascii	"<c\f\021\342\320\260\351\317\005\004K\267\345\214<"
	.ascii	"\376\n\247%\354Z+\353\273\252O\354\n\315\3675"
	.ascii	"\004\320\331T\025\300\031\371\275\261G\270\373\3166\344"
	.ascii	"\3604\252\"\037%\005\000\300\356(\317\b\245\366\256"
	.ascii	"\323\376\325\357\013\301.\317\371\274G\353\373Yo)"
	.ascii	"\020H\351\317\266\313\036\316A)\264\230\305\201\351\257"
	.ascii	"a1\313\036E\356-\372\363\017\266/\034\365\326E"
	.ascii	",\274\310Y\310\006+\320\342\317\265\3140\0300\367"
	.ascii	"^B\352T,)=\316I\034\017{3J\343\230"
	.ascii	"M\277\257\003\311\233\262.\023\322\275\352\362\027D\366"
	.ascii	"'(\356\r\022]\004\027\273\013\365\026\342\230\370\003"
	.ascii	"\n\026\372L\337q)\334\343!HgF\033\3328"
	.ascii	"\274\261L\313\272\251\f\345\371\365\b\377\031'\341\220"
	.ascii	"\245\250\354\274\263h\253\003\n\301L2\344E\312\b"
	.ascii	"'\234\310S\353`\367\360\273\320.6\276\325\275\027"
	.ascii	"'\373C\276\032\272\304F\271B4\276\274\327Y6"
	.ascii	"\265c\377\336\302\260+\372\324F\000\031C\247cK"
	.ascii	"\367\237\355\250>\263\255\353\372\02137\264\316i\371"
	.ascii	":\302\355[\276\\\262\356G`\303\017\272\232\216\221"
	.ascii	"X\256T\337\":\273K\314(8\370\326t\372\241"
	.ascii	"\245\257\263\r\007L\306\016\305\373\023\321\027\034\273\212"
	.ascii	"\272\346\261\305Y.\266\344\335B\036\020\370\333M\215"
	.ascii	"\r\302!\034\341\005B\357\373$,\342\311Jho"
	.ascii	"\306K\336\364\341\006H\357\307\363\t6\035\000\035\275"
	.ascii	"\334\370\f\036\024*3>>W\033\370.\252\f\215"
	.size	.L__constant_3x3x8x16xi8, 1152

	.type	.L__constant_16xi32,@object     # @__constant_16xi32
	.p2align	6, 0x0
.L__constant_16xi32:
	.word	4294965746                      # 0xfffff9f2
	.word	4294964822                      # 0xfffff656
	.word	1465                            # 0x5b9
	.word	4294964775                      # 0xfffff627
	.word	2753                            # 0xac1
	.word	5854                            # 0x16de
	.word	4294961383                      # 0xffffe8e7
	.word	1243                            # 0x4db
	.word	4294962829                      # 0xffffee8d
	.word	4294959881                      # 0xffffe309
	.word	4294961425                      # 0xffffe911
	.word	4294959360                      # 0xffffe100
	.word	4294965885                      # 0xfffffa7d
	.word	13854                           # 0x361e
	.word	4294961616                      # 0xffffe9d0
	.word	4294966738                      # 0xfffffdd2
	.size	.L__constant_16xi32, 64

	.type	.L__constant_144x16xi8,@object  # @__constant_144x16xi8
	.p2align	6, 0x0
.L__constant_144x16xi8:
	.ascii	"\227 \320\037\035\341\363E\241\274\257\262\336\026\277!"
	.ascii	"\331%\234\372\331\253\0049I\357\315\347\226\347\t\257"
	.ascii	"\327\032\335,\005\016\342\331V\f\256\273[\006\357*"
	.ascii	"\264\3056\005\3450}\023\307\311R\271\255\035=\317"
	.ascii	"m?\246\000I\247\f\370\021\237\355\323\222I\307\261"
	.ascii	"\325\025P7(\0331F\307lNL\262\034\346\035"
	.ascii	"\260\332\310@\004\377\031N(\263\020/\316\0227\354"
	.ascii	"\205\022\332\034J\004\305\362\006$\373\331(\343\375\005"
	.ascii	"\363\275\306\007\016\362\267\034\f\352\001\006C\324\377\336"
	.ascii	"\335\0338(I\343\"\265\355\027\033\274\301\3432\025"
	.ascii	"-\376V\026 \275\021\375\354\374\323\322P\037\314\266"
	.ascii	"w\311\374\306P\355\347P:B\000\nK\274\316."
	.ascii	"\214\024\311%H\267\334\362\001\2212\3444\013\336\336"
	.ascii	"kJ\254\270\266\\#\311\254!D\006\311Y\274\021"
	.ascii	"P\332\357>E\343\222\250\032\036\330\335\364\371L\304"
	.ascii	"h\002\344\276Y)b\370\365\026\270\310d\256\324\355"
	.ascii	"\002\334\356\366\265 Q\375\246\311\325\256\031\320,R"
	.ascii	"\301\331IRO\310j\271!j\336\354\035\313\0365"
	.ascii	"b6*\362\323W;\361\365\017\r[$\341\001E"
	.ascii	"\352\261^\341\316*>\037\302h\314\r\004\376\347I"
	.ascii	"\223#\320\271\322@\324\300\344\232\006'l\325\370\020"
	.ascii	"A\005+\355\353&\357\243\003\245\005\350\311#\331\356"
	.ascii	"xQ<\016\326\035N9<P$\t\241\331\013\315"
	.ascii	"6/\326\341J(\316\020YW\356\327\342\3007\373"
	.ascii	"?D\244\327\000%\314\357\250(M\026\241\030\323\036"
	.ascii	">\367\347\315\277\333\236\256\024,\257\003\027\270<@"
	.ascii	"\214\343\361\355\267M\323[\235\265W\355C\314\374\247"
	.ascii	"9\031`\367\254&\224\005\353\313\315\372QC\273\375"
	.ascii	"\r\007%\322T\311C\313\332\357\000\322m\250\r\323"
	.ascii	"\360J\265(%\243\254$\234\273\322\377\321N\032\325"
	.ascii	"U\330\254\0234G\252#\022\003\301\023(\0350\247"
	.ascii	"\336:+B\325\322@\016\036\341\257\250w>\013\307"
	.ascii	"bD\345\274\017\032\244\325:\031\235T3M\317\003"
	.ascii	"i\303Q\000/\244\311\321\275\030\246\343D#\260Y"
	.ascii	"\247\254\322\004\252X1\"\247\273\236\330f\337\335\004"
	.ascii	"\3168\336\025\360\033\026\352\246\310\025\356\3325\023\350"
	.ascii	"\373\325T\306\372\366\363\362\3620\337\330\247\007\364\363"
	.ascii	"\341J\311\366NG \373\027\362\323\"u2\017\030"
	.ascii	"\365K\35394\003\240R\352\317>\262b\022\271\001"
	.ascii	">Q\353\3131(\253\263\364\036\331\275\310S\342Y"
	.ascii	"\253\320\343(\275\026\234\242T\246\003\002l\334&9"
	.ascii	"\236\025\330O/\304r\277\237\273\031\362\212<\350\357"
	.ascii	"\272\314UO\347\340_Ka\242\237\370\227\367%\004"
	.ascii	"\220\335\273\311,\270\274\301\007\013\032\030@\302\fW"
	.ascii	"k:\325@\006\033hK(\001\035\251E\022\332\022"
	.ascii	"i\004\254\341\033\3177\031 \n\261 \234*\321V"
	.ascii	"\243\254\307\341M4\301\315\373\240\314\033s\246\271A"
	.ascii	"\251\336\276\332\317X\023\")GF\020\337\360CM"
	.ascii	"\225\274\250!M\311\252\020\305/C@\t\331\367\264"
	.ascii	"4\264D\276\274\030\335[\030\017\366,\355\315\005\026"
	.ascii	"\247\304\364\367\320\371\274\364\372S\0336\036\351\354\251"
	.ascii	"e\366\005\275\340\332S\332\377\23335\bL\004\355"
	.ascii	"I\343\"LY\256\375\313a\363\333?\351\335\2665"
	.ascii	"5\341O;\315\343 \305-=\350\360ZU\026\311"
	.ascii	"\255O\251\264\312\003S\344\372\273\026\2439,\316\344"
	.ascii	"\262\313\333 %\256\226L\253\243<\017b\345!\336"
	.ascii	"?6\243\016\026X\225B8\270\347\023QF7\376"
	.ascii	"1T?;/F\201\350\316\017=\3219\031\320\016"
	.ascii	"\216=\000\340\b\324PH1\235\n\321\r\360\3451"
	.ascii	"?+&\325\250\000\\O\314\3774\374\353&\337\312"
	.ascii	"\245\335\030PL\026P,\247\333\261>\357W\375\277"
	.ascii	"\2142\025\342[<g$/B\242%\343\336*\345"
	.ascii	"\362\b*\3340\021\372\3705,_\325\343\254\375M"
	.ascii	"U\322\n\352\006\277 @\367m\341\356/\330\327\306"
	.ascii	"\310\364\3566?\t\223\262I\324c\b \025\013\347"
	.ascii	"\245\004\337\372P\267T\023\344\362\0072B\266(\357"
	.ascii	"\331?\343\023\356\243T7\007;\243\314@@\027\247"
	.ascii	"\206\341\024\0314\377\313\033\0371 [\241\033\310\261"
	.ascii	"7&a> Km1\375\275\251H>\272\021\327"
	.ascii	"p\352\t\000\343\313|\257\243f\365\341\tE\022\270"
	.ascii	"\310\370[\305\350\246j\306P\026.\323H\035\0278"
	.ascii	"\037-7\017\350\255f\273\322\336\\\363(D\331\363"
	.ascii	"F\257\355\270\274Uq]W3\371\332\000\256\0341"
	.ascii	"\035\311W\017B\345\r7%0\f1Q<\263\035"
	.ascii	"\247\372\365\266\013\276\027\367\304\262\237\352\327S\313\364"
	.ascii	"0\370Z*\247\335\374\020\2534\000\364\225\337\000\""
	.ascii	")\036c\024\3263_\033\315\267\261\374\020\032\366Q"
	.ascii	"\275\253\024\277\332\361\223\366\354\277$\303\364\253\323K"
	.ascii	"\022:\243.\003\016\027\032aD\355:I\000\267\337"
	.ascii	"U\323)\000\035,\236\247\277F\236\262w\035\b\251"
	.ascii	"n\002\037\362P\270\262\347\037\327:\261x\346\321\033"
	.ascii	"\024R4)\377\037\340\361\330F\033\024\004#\3142"
	.ascii	"\002\024\242\003F\341\370\263C\251\272\001K\033\262\354"
	.ascii	"\260\352d-\322\f\275\347\371\274\363\025I\036\341\257"
	.ascii	"f\341J.\321\315r\302X\375\0069\3320,\033"
	.ascii	"tIB\312\"\371\206\t\005\261\001\324\n\2610\266"
	.ascii	"\273M\376\261\262\255\301\302\257_5V\fB\034\266"
	.ascii	"\\\034\355\3051\267\360\\\\\026\256\005\3179\377\027"
	.ascii	"\264\361\002\313-\304\236/,\222 L\317\350\261J"
	.ascii	"O\373\244%\027\335\365\331\373`\367\370Z\312\037\376"
	.ascii	"\310P\306\331\337\323\335\002\267\320\375\375\307\337\307Q"
	.ascii	"a\007\357\302\306.\3702\275\263V \215Z\3639"
	.ascii	"7\321\"K\333\362\216\026&\262\372\316\264\000J\370"
	.ascii	"\345\316\325\"!\316k\364\247 \246(\311\314@\371"
	.ascii	" \"\307\017P1\333\rT\327S\250\371\342>\324"
	.ascii	"\331\371\371\361/\2744\331\264\334\\\000$\356J\334"
	.ascii	"\353,\327'2\344\003\032\031X \350\327-,0"
	.ascii	"\377\360\242\345\374\276z1\365\311\272\364[\311\024\345"
	.ascii	"\"\373\357\316D\334\340\322\003\230\353 \024\324>1"
	.ascii	"\232\322\\\367\016\320\333\331a\356\241\375\2225\343\006"
	.ascii	"\375\026\0019\310!\236K\r\232\365\001kID$"
	.ascii	"\216\273*\300JJ\320\322\346/\357<\232\334+\307"
	.ascii	"3\333QB\247\b\026N\273\266^\276\257\365L\337"
	.ascii	"\022-L\312-\002\374\333\352\2426(\013\3740K"
	.ascii	"f\001\372\022\311\000\356\\\336M\031\366\030\016\306\320"
	.ascii	"\224\340\352\0353@\342\374\320\254M!\n\313*F"
	.ascii	"g0\337\264\244\266\0053a\262%*TXI\360"
	.ascii	"\205\305P'\364R2\256\365\005\300\264&\r\271\310"
	.ascii	"!\022\367\256\3641\323\002\253J\363\3328\355\333\343"
	.ascii	"\333\017\236\304\333:;\371\306\374\234FUXO-"
	.ascii	"@\330KE\311;\207=[\221\002\335\322?\361\017"
	.ascii	"\347\257\022\020&\036\201\254aN\002\360/+I\022"
	.ascii	"4E\374\037\370\333\306\372P\306\030\250R\001\307\246"
	.ascii	"\315\374\347\325\347\303\3158\321L\364\367\003\336\000\275"
	.ascii	"\004\017\253\017\310\354\375H\035\343\240\252\344\301:\304"
	.ascii	"\351\347\314\300\314\370\255\f\355\335I\3223\341\354\017"
	.ascii	"\254\027\302 \337^/%\333\036X+\313\317D-"
	.ascii	"\255\0065.Y\b\252DA.\036\315Z0\324\332"
	.ascii	"\223\343\262\3774/\277\321\007<\244311)\306"
	.ascii	"\346(\027' \304\301\326\324\023E'\346\265%D"
	.ascii	"{\017\3173\342\020\333\032\337\252\373\364\261\t6\354"
	.ascii	"\230J\367\317IWW\377\251\310\250OG\333\362\r"
	.ascii	"\347J D\350\311r\341\000\236\351V\357\310\035\022"
	.ascii	"M\264\002<\032\363\254\021\267h+\261\377\261\267\253"
	.ascii	"|\355\031\367\n\254\01762\334K\031\243\030LK"
	.ascii	"s\0006\357.\006\254\031\013=\315\006E\255\005?"
	.ascii	"\316\262\315\037X\331\361\f\336[\363\311I\032\3771"
	.ascii	"\244\265\264IOB\313\333\3457\275&,\027\360!"
	.ascii	"\2109\000\337\030S\034\366\243\255a g2\024F"
	.ascii	"\271\302\354D\347\320\377\360X\032\255\376\263\351E\300"
	.ascii	"\342\363:\002\374$Y\f\033JQ+\031\002\"\030"
	.ascii	"a\267\253\000\343\3647\027\277\367\250#O\020\2701"
	.ascii	"S\313b.\343\0041\260J5\337\253\360\"\307\341"
	.ascii	"\314R\250D\017B\302\266\"CS\360\356\3148\r"
	.ascii	"\214 #\332\006?\033\364\330\363\"C\016\035\035\013"
	.ascii	"\t\340\361?\375\342Y\363-\304\022\343\366\007\356\n"
	.ascii	"\317\007F\346\324R/\3711\376 \254NK\364\257"
	.ascii	"8I\334\377\022\374l\303@!\322\325)\006\261\247"
	.ascii	"HC,\333$\252\016\332\304\taE\023\333\276Q"
	.ascii	"f \265\000\271\320\3679cP9\000\264\020\301\035"
	.ascii	"f;\306\020\277\031\224\353\b\371I\274S\330\302\031"
	.ascii	";\275\307%\373\004z\270\004J\253/\226\251\021\275"
	.ascii	"\332\274\326B[\366a\027\336\346\343\311\225\261\316\026"
	.ascii	"\230S-\317\351\013\3369\362\013\261\267\317\331L#"
	.size	.L__constant_144x16xi8, 2304

	.type	.Lglobal_seed,@object           # @global_seed
	.local	.Lglobal_seed
	.comm	.Lglobal_seed,8,8
	.section	".note.GNU-stack","",@progbits
