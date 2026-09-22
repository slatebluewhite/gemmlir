"""A torchvision classifier with its *real* weights, on *real* images.

The model set this project validates against is exported with random weights at
64x64: that is the right thing for checking a compiler -- a wrong rewrite moves
the L2 whatever the weights are -- but it cannot classify anything. This script
is the other half: pretrained weights, 224x224, real photographs, and the
class names carried into the binary so the board prints what it sees.

Calibration reads every demo image, so the activation scales are measured on
the distribution the demo actually runs.
"""
import sys, os, glob, warnings, subprocess
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/jaemin/05_gemmlir/gemmlir/scripts")
import torch, torch_mlir, torchvision, torchvision.models as tvm
from torchvision import transforms
from PIL import Image
from calibrate import annotate, activation_ranges, normalize

OPT = "/home/jaemin/05_gemmlir/gemmlir/build/bin/gemmlir-opt"
FRONT = ["--split-grouped-conv", "--canonicalize",
         "--raise-spatial-sum-to-pool", "--canonicalize",
         "--conv-nchw-to-nhwc", "--canonicalize",
         "--average-pool-to-contraction", "--canonicalize"]
TRACE = {"googlenet", "inception_v3", "densenet121"}
HERE = os.path.dirname(os.path.abspath(__file__))

name = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
res = int(sys.argv[2]) if len(sys.argv) > 2 else 224

WEIGHTS = {  # torchvision's own enum, so the right preprocessing comes with it
    "resnet18": "ResNet18_Weights", "resnet50": "ResNet50_Weights",
    "mobilenet_v2": "MobileNet_V2_Weights", "squeezenet1_1": "SqueezeNet1_1_Weights",
    "googlenet": "GoogLeNet_Weights", "densenet121": "DenseNet121_Weights",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    "shufflenet_v2_x0_5": "ShuffleNet_V2_X0_5_Weights",
    "mnasnet0_5": "MNASNet0_5_Weights", "regnet_y_400mf": "RegNet_Y_400MF_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
}

def build():
    """A fresh instance with the same pretrained weights -- `torch_mlir.compile`
    leaves a compiled `forward` on the TorchScript class, so the second export
    attempt needs its own object."""
    w = getattr(tvm, WEIGHTS[name]).DEFAULT
    # GoogLeNet's and Inception's pretrained weights insist the auxiliary heads
    # exist, so they are built and then removed: they are training-only, and
    # leaving them in puts a `torch.jit.is_scripting()` branch -- and with it an
    # unlowerable `aten::warn` -- into the graph.
    m = getattr(tvm, name)(weights=w).eval()
    if hasattr(m, "aux_logits"):
        m.aux_logits = False
        m.aux1 = m.aux2 = None
    for parent in m.modules():
        for n, child in list(parent.named_children()):
            if isinstance(child, torchvision.ops.StochasticDepth):
                setattr(parent, n, torch.nn.Identity())
    return m

tf = transforms.Compose([
    transforms.Resize(res + 32), transforms.CenterCrop(res), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

paths = sorted(glob.glob(os.path.join(HERE, "img", "*.jpg")))
imgs = [tf(Image.open(p).convert("RGB")).unsqueeze(0) for p in paths]
names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
assert imgs, "no images in demo/img"
print("%s at %dx%d, %d images" % (name, res, res, len(imgs)))

raw = normalize(str(torch_mlir.compile(
    build(), imgs[0], output_type="linalg-on-tensors", use_tracing=name in TRACE)))
open(os.path.join(HERE, "%s_demo_raw.mlir" % name), "w").write(raw)
out = subprocess.run([OPT] + FRONT + [os.path.join(HERE, "%s_demo_raw.mlir" % name)],
                     capture_output=True, text=True)
if out.returncode:
    print("FRONTEND FAILED\n" + out.stderr[:800]); sys.exit(1)

m = build()
ranges = activation_ranges(m, imgs)
cal = annotate(out.stdout, ranges)
open(os.path.join(HERE, "%s_demo_cal.mlir" % name), "w").write(cal)

labels = open(os.path.join(HERE, "imagenet_classes.txt")).read().splitlines()
with torch.no_grad():
    ys = [m(x) for x in imgs]

def cstr(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

with open(os.path.join(HERE, "%s_demo_data.h" % name), "w") as f:
    f.write("/* %s, pretrained, %dx%d -- %d real photographs */\n"
            % (name, res, res, len(imgs)))
    f.write("#define NIMG %d\n#define NPIX %d\n#define NCLS %d\n"
            % (len(imgs), imgs[0].numel(), ys[0].numel()))
    f.write("static const char *img_name[NIMG] = {%s};\n"
            % ",".join(cstr(n) for n in names))
    f.write("static const float torch_x[NIMG][NPIX] = {\n")
    for x in imgs:
        f.write("{" + ",".join("%.9ef" % v for v in x.flatten()) + "},\n")
    f.write("};\n")
    f.write("static const float torch_y[NIMG][NCLS] = {\n")
    for y in ys:
        f.write("{" + ",".join("%.9ef" % v for v in y.flatten()) + "},\n")
    f.write("};\n")
    f.write("static const char *class_name[NCLS] = {\n")
    f.write(",".join(cstr(l) for l in labels) + "};\n")

for n, y in zip(names, ys):
    top = y.softmax(1)[0].topk(5)
    print("  %-10s %s" % (n, ", ".join("%s %.1f%%" % (labels[i], v * 100)
                                       for v, i in zip(top.values, top.indices))))
print("%s ok: %d contractions" % (name, len(ranges)))
