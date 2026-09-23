"""A decoder-only transformer -- the shape an LLM actually is -- at a size the
board can hold.

Written out rather than assembled from `nn.MultiheadAttention`, for the reason
`runvit.py` records: that module is opaque to the calibration and reaches
torch-mlir as one fused `tm_tensor.attention`. Every extent here comes from a
Python integer, never a shape read, because under tracing a shape read is a
tensor.

The causal mask is a large negative constant rather than -inf: the mask is added
in the dequantized domain and calibration measures the range of what it
produces, so an infinity there would take the scale of every attention score
with it.
"""
import sys, os, warnings, subprocess; warnings.filterwarnings("ignore")
import os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import torch, torch.nn as nn, torch_mlir
from calibrate import annotate, activation_ranges, normalize

OPT = os.environ.get("GEMMLIR_OPT", os.path.join(REPO, "build", "bin", "gemmlir-opt"))
FRONT = ["--split-grouped-conv", "--canonicalize",
         "--raise-spatial-sum-to-pool", "--canonicalize",
         "--conv-nchw-to-nhwc", "--canonicalize",
         "--average-pool-to-contraction", "--canonicalize"]

name  = sys.argv[1] if len(sys.argv) > 1 else "gpt_tiny"
T     = int(sys.argv[2]) if len(sys.argv) > 2 else 64      # context length
L     = int(sys.argv[3]) if len(sys.argv) > 3 else 4       # blocks
V, D, H, F = 512, 192, 3, 768


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, F)
        self.fc2 = nn.Linear(F, D)

    def forward(self, x, mask):
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.unflatten(-1, (H, D // H)).transpose(1, 2)
        k = k.unflatten(-1, (H, D // H)).transpose(1, 2)
        v = v.unflatten(-1, (H, D // H)).transpose(1, 2)
        a = torch.matmul(q, k.transpose(-2, -1)) * float((D // H) ** -0.5) + mask
        o = torch.matmul(torch.softmax(a, dim=-1), v)
        x = x + self.proj(o.transpose(1, 2).flatten(-2))
        h = self.ln2(x)
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(h)))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(1, T, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        m = torch.full((T, T), -1.0e4)
        self.register_buffer("mask", torch.triu(m, diagonal=1).reshape(1, 1, T, T))

    def forward(self, idx):
        x = self.tok(idx) + self.pos
        for b in self.blocks:
            x = b(x, self.mask)
        return self.head(self.lnf(x))


torch.manual_seed(11)
m = GPT().eval()
idx = torch.randint(0, V, (1, T))

try:
    raw = normalize(str(torch_mlir.compile(m, idx, output_type="linalg-on-tensors",
                                           use_tracing=True)))
except Exception as e:
    print("%-20s EXPORT FAILED\n%s" % (name, str(e)[:2000])); sys.exit(0)
open("%s_raw.mlir" % name, "w").write(raw)
out = subprocess.run([OPT] + FRONT + ["%s_raw.mlir" % name], capture_output=True, text=True)
if out.returncode:
    print("%-20s FRONTEND FAILED\n%s" % (name, out.stderr[:1200])); sys.exit(0)

ranges = activation_ranges(m, [idx])
try:
    cal = annotate(out.stdout, ranges)
except RuntimeError as e:
    print("%-20s CALIBRATION %s" % (name, str(e).split(';')[0])); sys.exit(0)
open("%s_cal.mlir" % name, "w").write(cal)
with torch.no_grad(): y = m(idx)
with open("%s_data.h" % name, "w") as f:
    f.write("static const int torch_idx[%d] = {%s};\n"
            % (idx.numel(), ",".join(str(int(v)) for v in idx.flatten())))
    f.write("static const float torch_y[%d] = {%s};\n"
            % (y.numel(), ",".join("%.9ef" % v for v in y.flatten())))
params = sum(p.numel() for p in m.parameters())
print("%-20s ok: %d contractions, %.2fM parameters, |y| max %.4f"
      % (name, len(ranges), params / 1e6, float(y.abs().max())))
