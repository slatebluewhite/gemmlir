"""Dump every stage between a PyTorch module and RISC-V assembly.

Small on purpose: two convolutions, a batch norm on each, a residual add and a
max-pool, on a 16x16 image with 8 channels in. Everything a real network does to
give this compiler trouble is here -- a norm that has to be folded, an
activation that has to be fused, a join the accelerator cannot write into, and a
pool -- and the dumps are still short enough to read.

    python3 walkthrough.py        # writes 01_*.mlir .. 10_*.s beside this file
"""
import os, subprocess, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/jaemin/05_gemmlir/gemmlir/scripts")
import torch, torch.nn as nn, torch_mlir
from calibrate import annotate, activation_ranges, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
G = "/home/jaemin/05_gemmlir/gemmlir"
OPT = f"{G}/build/bin/gemmlir-opt"
TRANSLATE = "/home/jaemin/05_gemmlir/llvm-project/build/bin/mlir-translate"
LLC = "/home/jaemin/05_gemmlir/llvm-project/build/bin/llc"

def w(name, text):
    open(os.path.join(HERE, name), "w").write(text if text.endswith("\n") else text + "\n")
    print(f"  {name}  ({len(text.splitlines())} lines)")

def opt(src, passes, name):
    r = subprocess.run([OPT] + passes, input=src, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[:1500]); sys.exit(1)
    w(name, r.stdout)
    return r.stdout


class Block(nn.Module):
    """conv-bn-relu, then conv-bn added back to it, then relu and a max-pool."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        return self.pool(torch.relu(h + self.bn2(self.conv2(h))))


def build():
    torch.manual_seed(0)
    m = Block().eval()
    for mod in m.modules():                 # give the norms something to fold
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.3)
            mod.running_var.uniform_(0.5, 1.5)
    return m

x = torch.randn(1, 8, 16, 16)
print("stages:")
w("00_model.py", open(__file__).read().split("class Block")[1].split("def build")[0]
  .join(["class Block", ""]))

raw = normalize(str(torch_mlir.compile(build(), x, output_type="linalg-on-tensors")))
w("01_torch_mlir.mlir", raw)

FRONT_SHAPE = ["--split-grouped-conv", "--canonicalize",
               "--raise-spatial-sum-to-pool", "--canonicalize",
               "--conv-nchw-to-nhwc", "--canonicalize",
               "--average-pool-to-contraction", "--canonicalize"]
shaped = opt(raw, FRONT_SHAPE, "02_nhwc.mlir")

cal = annotate(shaped, activation_ranges(build(), [x]))
w("03_calibrated.mlir", cal)

def array(name, span):
    """The pass list as compile.sh has it -- read from the script, not retyped,
    so this walkthrough cannot drift from the pipeline it is documenting."""
    out = subprocess.run(["bash", "-c",
        f'cd {G}; dataflow=ws; quantize=1; eval "$(sed -n \'{span}\' scripts/compile.sh)"; '
        f'printf "%s\\n" "${{{name}[@]}}"'], capture_output=True, text=True).stdout.split("\n")
    return [a for a in out if a.strip()]

# From CONVERT=, not from FRONT=: the array holds "$CONVERT" and "$FLUSH", and
# starting the range below them leaves the accelerator conversion an empty
# string -- which looks exactly like a pipeline that converts nothing.
FRONT = array("FRONT", "66,128p")
MID = array("MID", "/^MID=(/,/convert-scf-to-cf)$/p")
LOWER = array("LOWER", "/^LOWER=(/,/--set-target-data-layout)$/p")

cut = next(i for i, a in enumerate(FRONT) if a.startswith("--one-shot-bufferize"))
quant = opt(cal, FRONT[:cut], "04_quantized.mlir")
buf = opt(quant, FRONT[cut:cut + 2], "05_memref.mlir")
accel = opt(buf, FRONT[cut + 2:], "06_accelerator.mlir")
host = opt(accel, MID, "07_host_loops.mlir")
llvmd = opt(host, LOWER, "08_llvm_dialect.mlir")

r = subprocess.run([TRANSLATE, "--mlir-to-llvmir"], input=llvmd, capture_output=True, text=True)
w("09_llvm.ll", r.stdout)
# The input and what PyTorch answers for it, so the object can be run and
# checked rather than only read.
with torch.no_grad():
    y = build()(x)
with open(os.path.join(HERE, "11_data.h"), "w") as f:
    f.write("static const float torch_x[%d] = {%s};\n"
            % (x.numel(), ",".join("%.9ef" % v for v in x.flatten())))
    f.write("static const float torch_y[%d] = {%s};\n"
            % (y.numel(), ",".join("%.9ef" % v for v in y.flatten())))
print("  11_data.h  (input and PyTorch's answer)")

r2 = subprocess.run([LLC, "-O2", "-march=riscv64", "-mattr=+m,+a,+f,+d,+c",
                     "-target-abi=lp64d", "-filetype=asm"], input=r.stdout,
                    capture_output=True, text=True)
w("10_riscv.s", r2.stdout)
print("done")
