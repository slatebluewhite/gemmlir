; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @tiled_matmul_auto(i64, i64, i64, ptr, ptr, ptr, ptr, i64, i64, i64, i64, float, float, float, i32, float, float, i1, i1, i1, i1, i1, i8, i32)

define void @matmul_example(ptr %0, ptr %1, ptr %2) !dbg !3 {
  call void @tiled_matmul_auto(i64 128, i64 256, i64 128, ptr %0, ptr %1, ptr null, ptr %2, i64 128, i64 256, i64 256, i64 256, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, i32 0, float 1.000000e+00, float 1.000000e+00, i1 false, i1 false, i1 false, i1 true, i1 false, i8 0, i32 0), !dbg !7
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "matmul_example", linkageName: "matmul_example", scope: null, file: !4, line: 3, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "07_final.mlir", directory: ".")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 12, column: 5, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 13, column: 5, scope: !8)
