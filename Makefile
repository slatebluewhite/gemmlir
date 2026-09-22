# Thin wrappers around the commands in README. Override variables on the command
# line, or once and for all in an untracked `config.mk`:
#   make llvm LLVM_SRC=~/llvm-project JOBS=8
#   make build MLIR_DIR=/path/to/lib/cmake/mlir
#   echo 'LLVM_SRC = /somewhere/llvm-project' > config.mk
#   echo 'RISCV_CC = riscv64-linux-gnu-gcc'    >> config.mk
-include config.mk

RISCV_CC    ?=
LLVM_SRC    ?= $(HOME)/llvm-project
LLVM_COMMIT ?= 367e3889fabc
LLVM_BUILD  ?= $(LLVM_SRC)/build
MLIR_DIR    ?= $(LLVM_BUILD)/lib/cmake/mlir
LLVM_DIR    ?= $(LLVM_BUILD)/lib/cmake/llvm
LIT         ?= $(LLVM_BUILD)/bin/llvm-lit
BUILD       ?= build
JOBS        ?= $(shell nproc)

.PHONY: all build test example conv-example llvm apt-install clean

all: build

apt-install:                      ## host packages needed to build LLVM and gemmlir
	sudo apt-get install -y build-essential cmake ninja-build python3 git


llvm:                             ## clone, check out and build LLVM/MLIR with the RISCV target (~1 h)
	test -d $(LLVM_SRC)/.git || git clone https://github.com/llvm/llvm-project.git $(LLVM_SRC)
	git -C $(LLVM_SRC) checkout $(LLVM_COMMIT)
	cmake -G Ninja -S $(LLVM_SRC)/llvm -B $(LLVM_BUILD) -DCMAKE_BUILD_TYPE=Release \
	  -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
	  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON
	ninja -C $(LLVM_BUILD) -j$(JOBS)

build:                            ## configure and build gemmlir-opt
	cmake -G Ninja -S . -B $(BUILD) -DMLIR_DIR=$(MLIR_DIR) -DLLVM_DIR=$(LLVM_DIR) -DLLVM_EXTERNAL_LIT=$(LIT) \
	  $(if $(RISCV_CC),-DGEMMLIR_RISCV_CC=$(shell command -v $(RISCV_CC) || echo $(RISCV_CC)),)
	ninja -C $(BUILD) -j$(JOBS) gemmlir-opt

test: build                       ## lit tests
	ninja -C $(BUILD) check-gemmlir

example: build                    ## lower examples/matmul_i8.mlir to $(BUILD)/matmul.o
	GEMMLIR_BUILD=$(BUILD) LLVM_BIN=$(LLVM_BUILD)/bin ./scripts/compile.sh examples/matmul_i8.mlir -o $(BUILD)/matmul.o

conv-example: build               ## lower examples/conv_i8.mlir to $(BUILD)/conv.o
	GEMMLIR_BUILD=$(BUILD) LLVM_BIN=$(LLVM_BUILD)/bin ./scripts/compile.sh examples/conv_i8.mlir -o $(BUILD)/conv.o

clean:
	rm -rf $(BUILD)
