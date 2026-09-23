"""One torchvision model, one model instance: export, calibrate and reference
output all from the same weights.

Splitting this across two processes is how the L2 numbers came out meaningless
the first time -- each process drew its own random initialisation, so the
compiled network and the network the reference came from were different.
"""
import sys, os, time, copy, warnings, subprocess; warnings.filterwarnings("ignore")
import os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import torch, torch_mlir, torchvision.models as tvm, torchvision.ops
from calibrate import annotate, activation_ranges, normalize

OPT = os.environ.get("GEMMLIR_OPT", os.path.join(REPO, "build", "bin", "gemmlir-opt"))
FRONT = ["--split-grouped-conv", "--canonicalize",
         "--raise-spatial-sum-to-pool", "--canonicalize",
         "--conv-nchw-to-nhwc", "--canonicalize",
         "--average-pool-to-contraction", "--canonicalize"]


def drop_stochastic_depth(model):
    """`StochasticDepth` returns its input untouched in eval mode; the only
    thing that reaches torch-mlir is its argument validation, a Python `in` on
    a string, which arrives as `aten.__contains__.str_list` and has no
    lowering. Swapping it for `nn.Identity` is exact for inference."""
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, torchvision.ops.StochasticDepth):
                setattr(parent, name, torch.nn.Identity())
    return model


# Inference configuration where torchvision's constructor has one. GoogLeNet's
# and Inception v3's auxiliary classifiers are training-only -- torchvision's own
# pretrained weights are loaded with them off -- and leaving them on is what puts
# the `torch.jit.is_scripting()` branch, and with it a `warnings.warn` that
# arrives as an unlowerable `aten::warn`, into the graph.
# Models that must be **traced** rather than scripted. Scripting keeps branches
# the model takes only under `torch.jit.is_scripting()`, and GoogLeNet, Inception
# v3 and DenseNet all put something unlowerable there -- the first two a
# `warnings.warn` that arrives as `aten::warn`, DenseNet a `forward__0` its
# checkpointing wrapper defines. Tracing does not enter those branches, and for
# an inference graph with one fixed input shape the two agree. (ViT is traced
# too, for its own reason; see runvit.py.)
TRACE = {"googlenet", "inception_v3", "densenet121"}

INFERENCE_KWARGS = {
    "googlenet": dict(init_weights=False, aux_logits=False),
    "inception_v3": dict(init_weights=False, aux_logits=False),
}

def build():
    """A fresh model with the same weights. Both attempts below need one of
    their own: a failed `torch_mlir.compile` leaves a compiled `forward` behind
    on the TorchScript class, and the next `compile` of that class stops with
    "method 'X.forward' already defined" -- a `deepcopy` does not help, because
    the class is what carries it."""
    torch.manual_seed(11)
    model = drop_stochastic_depth(
        getattr(tvm, name)(**INFERENCE_KWARGS.get(name, {})).eval())
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            mod.running_mean.normal_(0, 0.3); mod.running_var.uniform_(0.5, 1.5)
    return model


def export(sample):
    """One attempt, in the mode this model needs -- there is no retrying.

    A `torch_mlir.compile` registers the model's TorchScript class in the
    process-wide compilation unit, and it stays there whether the compile
    succeeded or not: a second attempt stops with "method 'X.forward' already
    defined" no matter how fresh the instance is. So the mode is a property of
    the model, recorded in `TRACE`, not something to discover at runtime.
    """
    return normalize(str(torch_mlir.compile(
        build(), sample, output_type="linalg-on-tensors",
        use_tracing=name in TRACE))), build()


name, res = sys.argv[1], int(sys.argv[2])
torch.manual_seed(11)
x = torch.randn(1, 3, res, res)

try:
    raw, m = export(x)
except Exception as e:
    msg = str(e)
    tail = msg.split("traced:")[-1] if "traced:" in msg else msg
    bad = [l for l in tail.splitlines() if "aten." in l or "error:" in l]
    print("%-20s EXPORT FAILED %s" % (name, (bad or [tail.strip()[:90]])[0][:110]))
    sys.exit(0)
open("%s_raw.mlir" % name, "w").write(raw)
out = subprocess.run([OPT] + FRONT + ["%s_raw.mlir" % name], capture_output=True, text=True)
if out.returncode:
    print("%-20s FRONTEND FAILED" % name); sys.exit(0)

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
