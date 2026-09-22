"""The four MLPerf Tiny v1.0 benchmarks, as PyTorch modules.

Architectures follow the reference implementations in mlcommons/tiny
(benchmark/training/*): resnet_v1_eembc for image classification, the DS-CNN of
keyword_spotting, MobileNetV1 0.25x for visual wake words, and the dense
autoencoder of anomaly_detection. Written with `bias=False` + BatchNorm
throughout, which is the shape the frontend recipe wants.
"""
import torch, torch.nn as nn


def cbr(i, o, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(i, o, k, stride=s, padding=p, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU())


class ResNet8(nn.Module):
    """MLPerf Tiny image classification: ResNet-8 on CIFAR-10, 32x32x3 -> 10."""
    def __init__(self):
        super().__init__()
        self.stem = cbr(3, 16)
        self.a1 = cbr(16, 16); self.a2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16))
        self.b1 = cbr(16, 32, s=2); self.b2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32))
        self.bsc = nn.Sequential(nn.Conv2d(16, 32, 1, stride=2, bias=False),
                                 nn.BatchNorm2d(32))
        self.c1 = cbr(32, 64, s=2); self.c2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64))
        self.csc = nn.Sequential(nn.Conv2d(32, 64, 1, stride=2, bias=False),
                                 nn.BatchNorm2d(64))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem(x)
        x = torch.relu(self.a2(self.a1(x)) + x)
        x = torch.relu(self.b2(self.b1(x)) + self.bsc(x))
        x = torch.relu(self.c2(self.c1(x)) + self.csc(x))
        return self.fc(x.mean(dim=(2, 3)))


class DSCNN(nn.Module):
    """MLPerf Tiny keyword spotting: DS-CNN on a 49x10 MFCC, -> 12 words."""
    def __init__(self, ch=64):
        super().__init__()
        # A 10x4 kernel: not square, so the accelerator's convolution cannot
        # express it and --conv-to-img2col packs it as a matmul.
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch, (10, 4), stride=(2, 2), padding=(5, 1), bias=False),
            nn.BatchNorm2d(ch), nn.ReLU())
        blocks = []
        for _ in range(4):
            blocks += [nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
                       nn.BatchNorm2d(ch), nn.ReLU(),
                       nn.Conv2d(ch, ch, 1, bias=False),
                       nn.BatchNorm2d(ch), nn.ReLU()]
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(ch, 12)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.fc(x.mean(dim=(2, 3)))


class MobileNetV1(nn.Module):
    """MLPerf Tiny visual wake words: MobileNetV1 0.25x on 96x96x3 -> 2."""
    def __init__(self, a=0.25):
        super().__init__()
        def c(n): return max(8, int(n * a))
        plan = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
                (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2),
                (1024, 1)]
        layers = [cbr(3, c(32), s=2)]
        prev = c(32)
        for out, s in plan:
            layers += [nn.Conv2d(prev, prev, 3, stride=s, padding=1,
                                 groups=prev, bias=False),
                       nn.BatchNorm2d(prev), nn.ReLU(),
                       nn.Conv2d(prev, c(out), 1, bias=False),
                       nn.BatchNorm2d(c(out)), nn.ReLU()]
            prev = c(out)
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, 2)

    def forward(self, x):
        return self.fc(self.body(x).mean(dim=(2, 3)))


class AutoEncoder(nn.Module):
    """MLPerf Tiny anomaly detection: a dense autoencoder, 640 -> 640."""
    def __init__(self, d=640, h=128, z=8):
        super().__init__()
        def dbr(i, o):
            return nn.Sequential(nn.Linear(i, o, bias=False), nn.BatchNorm1d(o),
                                 nn.ReLU())
        self.net = nn.Sequential(dbr(d, h), dbr(h, h), dbr(h, h), dbr(h, h),
                                 dbr(h, z), dbr(z, h), dbr(h, h), dbr(h, h),
                                 dbr(h, h), nn.Linear(h, d))

    def forward(self, x):
        return self.net(x)


BENCH = {
    "tiny_ic":  (ResNet8,     lambda: torch.randn(1, 3, 32, 32)),
    "tiny_kws": (DSCNN,       lambda: torch.randn(1, 1, 49, 10)),
    "tiny_vww": (MobileNetV1, lambda: torch.randn(1, 3, 96, 96)),
    "tiny_ad":  (AutoEncoder, lambda: torch.randn(1, 640)),
}
