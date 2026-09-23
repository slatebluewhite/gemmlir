"""torchvision's own VisionTransformer, at a size the board can hold.

Same class the library ships `vit_b_16` from, at DeiT-Tiny geometry: the point
is the shape -- a patch-embedding convolution, a class token, a learned position
embedding, twelve blocks of multi-head attention and an MLP, layer norm
throughout -- not the parameter count. `vit_b_16` is 86M parameters and its
binary will not run over an NFS root.
"""
import sys, os, warnings, subprocess; warnings.filterwarnings("ignore")
import os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import torch, torch_mlir
from torchvision.models import VisionTransformer
from calibrate import annotate, activation_ranges, normalize

# `nn.MultiheadAttention` calls `F.scaled_dot_product_attention`, which torch-mlir
# lowers to `tm_tensor.attention` -- one fused op from its own TMTensor dialect,
# which nothing downstream of us has. Substituting SDPA's own math definition
# leaves the arithmetic alone and gives the two matmuls and the softmax the
# accelerator can actually see. It stays patched for the reference output too,
# so the numbers being compared come from exactly what was compiled.
# `nn.MultiheadAttention` is opaque twice over. Traced, it reaches torch-mlir as
# `tm_tensor.attention`, one fused op from torch-mlir's own TMTensor dialect that
# nothing downstream has. And `F.multi_head_attention_forward` dispatches through
# `has_torch_function`, which a `TorchFunctionMode` answers -- so the calibration
# mode is handed the whole call and everything inside it runs with the mode off:
# the two attention matmuls and the packed QKV projection were invisible, 48 of
# the model's 74 contractions.
#
# Written out, it is the same arithmetic with nothing hidden. Every extent comes
# from the module's own Python integers (`unflatten`/`flatten`, never a shape
# read) because under tracing a shape read is a tensor, and `** -0.5` on one
# becomes an `aten::pow` that lowers to an `arith.fptosi` returning `si64`, which
# does not verify. `out_proj` goes through its module so the ordinary forward
# hook records it.
def _mha_forward(self, query, key, value, key_padding_mask=None,
                 need_weights=True, attn_mask=None, average_attn_weights=True,
                 is_causal=False):
    assert self.batch_first and query is key and key is value
    h, d = self.num_heads, self.head_dim
    qkv = torch.nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.unflatten(-1, (h, d)).transpose(1, 2)
    k = k.unflatten(-1, (h, d)).transpose(1, 2)
    v = v.unflatten(-1, (h, d)).transpose(1, 2)
    a = torch.matmul(q, k.transpose(-2, -1)) * float(d ** -0.5)
    if attn_mask is not None:
        a = a + attn_mask
    o = torch.matmul(torch.softmax(a, dim=-1), v).transpose(1, 2).flatten(-2)
    return self.out_proj(o), None


torch.nn.MultiheadAttention.forward = _mha_forward

OPT = os.environ.get("GEMMLIR_OPT", os.path.join(REPO, "build", "bin", "gemmlir-opt"))
FRONT = ["--split-grouped-conv", "--canonicalize",
         "--raise-spatial-sum-to-pool", "--canonicalize",
         "--conv-nchw-to-nhwc", "--canonicalize",
         "--average-pool-to-contraction", "--canonicalize"]

name = sys.argv[1] if len(sys.argv) > 1 else "vit_tiny"
res  = int(sys.argv[2]) if len(sys.argv) > 2 else 64
L    = int(sys.argv[3]) if len(sys.argv) > 3 else 12

torch.manual_seed(11)
m = VisionTransformer(image_size=res, patch_size=16, num_layers=L, num_heads=3,
                      hidden_dim=192, mlp_dim=768, num_classes=10).eval()
# torchvision initialises the classification head to **zeros**, so the model as
# constructed answers 0 for every input and a relative error against it is 0/0.
torch.nn.init.normal_(m.heads.head.weight, std=0.05)
torch.nn.init.normal_(m.heads.head.bias, std=0.05)
x = torch.randn(1, 3, res, res)

try:
    # Scripting a ViT stops at "unsupported by backend contract: module
    # initializers"; tracing gets through, and the shapes are fixed anyway.
    raw = normalize(str(torch_mlir.compile(m, x, output_type="linalg-on-tensors",
                                           use_tracing=True)))
except Exception as e:
    print("%-20s EXPORT FAILED\n%s" % (name, str(e)[:1500])); sys.exit(0)
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
