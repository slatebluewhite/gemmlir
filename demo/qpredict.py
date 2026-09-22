"""Will the int8 model still name the same class?

The board answers this for the four demo photographs. This answers it for more
of them, and says whether a per-channel weight scale -- the standard thing the
pipeline does not do -- would change the answer. Same rounding and saturation
the compiler uses; activations are left in f32, so what is measured is the
weight quantization alone.
"""
import sys, glob, os, warnings; warnings.filterwarnings("ignore")
import torch, torchvision.models as tvm
from torchvision import transforms
from PIL import Image

def q_per_tensor(w):
    s = w.abs().max() / 127.0
    if s == 0: return w
    return torch.clamp(torch.round(w / s), -128, 127) * s

def q_per_channel(w):
    flat = w.reshape(w.shape[0], -1)
    s = (flat.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
    shape = (-1,) + (1,) * (w.dim() - 1)
    return torch.clamp(torch.round(w / s.reshape(shape)), -128, 127) * s.reshape(shape)

def requantize(model, how):
    m = model
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                mod.weight.copy_(how(mod.weight))
    return m

MODELS = {"resnet18": "ResNet18_Weights", "mobilenet_v2": "MobileNet_V2_Weights",
          "squeezenet1_1": "SqueezeNet1_1_Weights", "googlenet": "GoogLeNet_Weights",
          "resnet50": "ResNet50_Weights"}
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
imgs = [tf(Image.open(p).convert("RGB")).unsqueeze(0) for p in sorted(glob.glob("img/*.jpg"))]
x = torch.cat(imgs)
print("%d images\n" % len(imgs))
for name, w in MODELS.items():
    def build():
        kw = {}
        m = getattr(tvm, name)(weights=getattr(tvm, w).DEFAULT, **kw).eval()
        if hasattr(m, "aux_logits"): m.aux_logits = False; m.aux1 = m.aux2 = None
        return m
    with torch.no_grad():
        ref = build()(x)
        pt = requantize(build(), q_per_tensor)(x)
        pc = requantize(build(), q_per_channel)(x)
    def report(y):
        l2 = float(((y - ref) ** 2).sum().sqrt() / (ref ** 2).sum().sqrt())
        agree = int((y.argmax(1) == ref.argmax(1)).sum())
        return l2, agree
    a, b = report(pt), report(pc)
    print("%-16s per-tensor  L2 %.4f  top-1 %d/%d     per-channel  L2 %.4f  top-1 %d/%d"
          % (name, a[0], a[1], len(imgs), b[0], b[1], len(imgs)))
