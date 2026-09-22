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


