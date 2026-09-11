# Thin wrappers around the commands in README. Override variables on the command line:
#   make llvm LLVM_SRC=~/llvm-project JOBS=8
#   make build MLIR_DIR=/path/to/lib/cmake/mlir
LLVM_SRC    ?= $(HOME)/llvm-project
LLVM_COMMIT ?= 367e3889fabc
LLVM_BUILD  ?= $(LLVM_SRC)/build
MLIR_DIR    ?= $(LLVM_BUILD)/lib/cmake/mlir
LLVM_DIR    ?= $(LLVM_BUILD)/lib/cmake/llvm
LIT         ?= $(LLVM_BUILD)/bin/llvm-lit
BUILD       ?= build
JOBS        ?= $(shell nproc)

.PHONY: all build test example llvm update-submodule apt-install clean

all: build

apt-install:                      ## host packages needed to build LLVM and gemmlir
	sudo apt-get install -y build-essential cmake ninja-build python3 git

update-submodule:                 ## gemmini-rocc-tests, headers only (include/ + rocc-software/src)
	git submodule update --init third_party/gemmini-rocc-tests
	git -C third_party/gemmini-rocc-tests sparse-checkout set include rocc-software
	git -C third_party/gemmini-rocc-tests submodule update --init rocc-software

llvm:                             ## clone, check out and build LLVM/MLIR with the RISCV target (~1 h)
	test -d $(LLVM_SRC)/.git || git clone https://github.com/llvm/llvm-project.git $(LLVM_SRC)
	git -C $(LLVM_SRC) checkout $(LLVM_COMMIT)
	cmake -G Ninja -S $(LLVM_SRC)/llvm -B $(LLVM_BUILD) -DCMAKE_BUILD_TYPE=Release \
	  -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
	  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON
	ninja -C $(LLVM_BUILD) -j$(JOBS)

build:                            ## configure and build gemmlir-opt
	cmake -G Ninja -S . -B $(BUILD) -DMLIR_DIR=$(MLIR_DIR) -DLLVM_DIR=$(LLVM_DIR) -DLLVM_EXTERNAL_LIT=$(LIT)
	ninja -C $(BUILD) -j$(JOBS) gemmlir-opt

test: build                       ## lit tests
	ninja -C $(BUILD) check-gemmlir

example: build                    ## lower examples/matmul_i8.mlir to matmul.o
	GEMMLIR_BUILD=$(BUILD) LLVM_BIN=$(LLVM_BUILD)/bin ./scripts/compile.sh examples/matmul_i8.mlir -o matmul.o

clean:
	rm -rf $(BUILD) matmul.o
