# -*- Python -*-
import os
import lit.formats
from lit.llvm import llvm_config

config.name = "GEMMLIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.gemmlir_obj_root, "test")
config.excludes = ["CMakeLists.txt", "Inputs"]

llvm_config.with_system_environment(["HOME", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

gemmlir_tools_dir = os.path.join(config.gemmlir_obj_root, "bin")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.add_tool_substitutions(
    ["gemmlir-opt", "mlir-opt", "mlir-translate"],
    [gemmlir_tools_dir, config.llvm_tools_dir])
