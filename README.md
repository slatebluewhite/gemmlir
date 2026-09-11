# gemmlir

## Abstract
gemmlir is an out-of-tree MLIR dialect that offloads `linalg.matmul` to the
[Gemmini](https://github.com/ucb-bar/gemmini) accelerator: `linalg.matmul` →
`gemmlir.matmul_i8` → `gemmini_flush` + call to `tiled_matmul_auto` (`gemmini.h`) →
RISC-V LLVM IR. Tiling is done by the Gemmini runtime. f32 tensor inputs can be
force-quantized to the int8 path first.

![pipeline](docs/img/gemmlir-pipeline.png)

Pass pipelines: [docs/pipeline.md](docs/pipeline.md).

## Prerequisites

```
LLVM/MLIR >= 22 (tested: llvm-project 367e3889fabc), built with the RISCV target
CMake >= 3.20, Ninja, Python 3
gemmini-rocc-tests (submodule, 7c540b3; headers only)
riscv64-unknown-linux-gnu-gcc, a Gemmini-enabled Rocket SoC running Linux   (to run)
```

## Installation

```bash
git clone <this repo> gemmlir && cd gemmlir
make apt-install          # cmake, ninja, python3, compiler
make update-submodule     # gemmini.h and friends (sparse checkout of gemmini-rocc-tests)
make llvm                 # clone + build llvm-project at 367e3889fabc into ~/llvm-project (~1 h)
make                      # gemmlir-opt into build/
make test                 # lit tests
```

Each target is a few lines in the `Makefile`; override `LLVM_SRC`, `MLIR_DIR`, `LLVM_DIR`, `BUILD`,
`JOBS` on the command line to use an existing LLVM or another location.
`-DGEMMLIR_GEMMINI_PARAMS=<file>` (cmake) selects the `gemmini_params.h` generated for
your hardware instead of the submodule default.

## Usage

```bash
make example              # = compile.sh examples/matmul_i8.mlir -o matmul.o

export LLVM_BIN=~/llvm-project/build/bin
./scripts/compile.sh examples/matmul_i8.mlir -o matmul.o                  # memref, i8 x i8 -> i32
./scripts/compile.sh examples/matmul_f32_tensor.mlir -o matmul.o --quantize   # tensor f32
./scripts/compile.sh examples/matmul_i8.mlir --emit=llvm-ir               # or gemmlir | llvm-dialect

CFLAGS="-O2 -march=rv64gc -mabi=lp64d -static"
riscv64-unknown-linux-gnu-gcc $CFLAGS -Ithird_party/gemmini-rocc-tests -c runtime/gemmlir_rt.c -o gemmlir_rt.o
riscv64-unknown-linux-gnu-gcc $CFLAGS examples/main.c matmul.o gemmlir_rt.o -o matmul
# copy `matmul` to the board and run it; prints PASS
```

`compile.sh` chains `gemmlir-opt`, `mlir-translate` and `llc`. The lowered function is plain C,
`void matmul_example(int8_t *A, int8_t *B, int32_t *C)` (row-major, fixed shapes);
`examples/main.c` calls it and checks against a CPU reference. `runtime/gemmlir_rt.c`
provides `tiled_matmul_auto` and static-asserts that `gemmini_params.h` matches the ABI
the compiler emits.

## Status
The original implementation was verified on a Gemmini-enabled Rocket SoC on FPGA
(vivado-risc-v, Linux). This repackaged MLIR 22 version is verified on x86 (build,
`check-gemmlir`, both `compile.sh` paths to LLVM IR / riscv64 object) but has not been
re-run on the board.
