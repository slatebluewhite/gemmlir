"""A recurrent model: two `nn.LSTMCell` layers unrolled over a sequence.

`nn.LSTM` cannot be exported at all -- it keeps its parameters in a Python list
attribute (`_flat_weights`) which torch-mlir globalizes and then refuses as a
module initializer, tracing or not. `nn.LSTMCell` gets one step further and stops
at `aten.unsafe_chunk`, which has no lowering. Written out, the cell is four
gates off two matmuls and nothing else.
"""
import sys, warnings, subprocess; warnings.filterwarnings("ignore")
import os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import torch, torch_mlir
import torch.nn.functional as F
from calibrate import annotate, activation_ranges, normalize

OPT = os.environ.get("GEMMLIR_OPT", os.path.join(REPO, "build", "bin", "gemmlir-opt"))
FRONT = ["--split-grouped-conv", "--canonicalize",
         "--raise-spatial-sum-to-pool", "--canonicalize",
         "--conv-nchw-to-nhwc", "--canonicalize",
         "--average-pool-to-contraction", "--canonicalize"]


def _cell_forward(self, x, hx):
    """`nn.LSTMCell`, by its own definition. `chunk` where the module uses
    `unsafe_chunk`; the two differ only in whether the views alias."""
    h, c = hx
    g = (F.linear(x, self.weight_ih, self.bias_ih)
         + F.linear(h, self.weight_hh, self.bias_hh))
    i, f, gg, o = g.chunk(4, dim=1)
    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
    return torch.sigmoid(o) * torch.tanh(c), c


torch.nn.LSTMCell.forward = _cell_forward

STEPS, IN, HID = 16, 32, 48


class Recurrent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.LSTMCell(IN, HID)
        self.c2 = torch.nn.LSTMCell(HID, HID)
        self.head = torch.nn.Linear(HID, 10)

    def forward(self, x):
        h1 = torch.zeros(1, HID); c1 = torch.zeros(1, HID)
        h2 = torch.zeros(1, HID); c2 = torch.zeros(1, HID)
        for t in range(STEPS):
            h1, c1 = self.c1(x[:, t], (h1, c1))
            h2, c2 = self.c2(h1, (h2, c2))
        return self.head(h2)


name = "lstm"
torch.manual_seed(11)
m = Recurrent().eval()
x = torch.randn(1, STEPS, IN)

try:
    raw = normalize(str(torch_mlir.compile(m, x, output_type="linalg-on-tensors",
                                           use_tracing=True)))
except Exception as e:
    print("%-20s EXPORT FAILED\n%s" % (name, str(e)[:1200])); sys.exit(0)
open("%s_raw.mlir" % name, "w").write(raw)
out = subprocess.run([OPT] + FRONT + ["%s_raw.mlir" % name], capture_output=True, text=True)
if out.returncode:
    print("%-20s FRONTEND FAILED\n%s" % (name, out.stderr[:800])); sys.exit(0)

ranges = activation_ranges(m, [x])
try:
    cal = annotate(out.stdout, ranges)
except RuntimeError as e:
    print("%-20s CALIBRATION %s" % (name, str(e).split(';')[0])); sys.exit(0)
open("%s_cal.mlir" % name, "w").write(cal)
with torch.no_grad(): y = m(x)
with open("%s_data.h" % name, "w") as f:
    f.write("static const float torch_x[%d] = {%s};\n"
            % (x.numel(), ",".join("%.9ef" % v for v in x.flatten())))
    f.write("static const float torch_y[%d] = {%s};\n"
            % (y.numel(), ",".join("%.9ef" % v for v in y.flatten())))
print("%-20s ok: %d contractions, |y| max %.4f" % (name, len(ranges), float(y.abs().max())))
