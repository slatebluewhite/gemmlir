# Reference output from the MLIR 15 version

Snapshots of every stage of the original (MLIR 15.0.7, typed-pointer) implementation,
kept as a behavioural reference for the MLIR 22 port. They were produced by the old
`Gemmlir-opt --gemmlir` / `--gemmllvm` passes, which correspond to today's
`--convert-linalg-to-gemmlir` / `--convert-gemmlir-to-llvm`.

Two differences from current output are expected and intentional:

- No `llvm.inline_asm` flush before the call. The `gemmini_flush(0)` was added after
  these snapshots were taken.
- `D_scale_factor` (14th argument) is `f32` here; the current lowering emits `i32`,
  matching `scale_acc_t` in the default `gemmini_params.h`.

`08_llvm.ll` was translated with LLVM 15 and uses typed pointers; feed it to a modern
`llc` with `--opaque-pointers` or regenerate it with `scripts/compile.sh --emit=llvm-ir`.
