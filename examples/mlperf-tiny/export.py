"""Lower one MLPerf Tiny benchmark to a calibrated _cal.mlir plus its data header."""
import sys, warnings, subprocess; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/jaemin/05_gemmlir/gemmlir/scripts"); sys.path.insert(0, ".")
import torch, torch_mlir
from calibrate import annotate, activation_ranges, normalize
import models

OPT = "/home/jaemin/05_gemmlir/gemmlir/build/bin/gemmlir-opt"
FRONT = ["--split-grouped-conv", "--canonicalize",
         "--raise-spatial-sum-to-pool", "--canonicalize",
         "--conv-nchw-to-nhwc", "--canonicalize",
         "--average-pool-to-contraction", "--canonicalize"]

name = sys.argv[1]
cls, mk = models.BENCH[name]
torch.manual_seed(7); m = cls().eval()
# Real batch-norm statistics: left at their defaults the layer folds to the
# identity and the model stops testing what it is here to test.
for mod in m.modules():
    if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        mod.running_mean.normal_(0, 0.3); mod.running_var.uniform_(0.5, 1.5)
x = mk()

ranges = activation_ranges(m, [x])
with torch.no_grad(): y = m(x)
raw = normalize(str(torch_mlir.compile(m, x, output_type="linalg-on-tensors")))
open("%s_raw.mlir" % name, "w").write(raw)
out = subprocess.run([OPT] + FRONT + ["%s_raw.mlir" % name],
                     capture_output=True, text=True)
if out.returncode:
    print(out.stderr[:3000], file=sys.stderr); sys.exit(1)
open("%s_cal.mlir" % name, "w").write(annotate(out.stdout, ranges))
with open("%s_data.h" % name, "w") as f:
    f.write("static const float torch_x[%d] = {%s};\n"
            % (x.numel(), ",".join("%.9ef" % v for v in x.flatten())))
    f.write("static const float torch_y[%d] = {%s};\n"
            % (y.numel(), ",".join("%.9ef" % v for v in y.flatten())))
print("%s: %d calibrated contractions" % (name, len(ranges)))
