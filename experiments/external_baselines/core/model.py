import torch
import torch.nn as nn
from torchvision.models import resnet18, efficientnet_b0, convnext_tiny


class WDCNN1D(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, _img, sig):
        if sig.dim() == 2:
            sig = sig.unsqueeze(1)
        x = self.features(sig)
        return self.head(x)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        b = out_ch // 4
        self.b1 = nn.Conv1d(in_ch, b, kernel_size=1, padding=0)
        self.b3 = nn.Conv1d(in_ch, b, kernel_size=3, padding=1)
        self.b5 = nn.Conv1d(in_ch, b, kernel_size=5, padding=2)
        self.b7 = nn.Conv1d(in_ch, b, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.b7(x)], dim=1)
        return self.act(self.bn(x))


class InceptionTime1D(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        blocks = []
        ch = 64
        for _ in range(6):
            blocks.append(InceptionBlock1D(ch, 128))
            ch = 128
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, _img, sig):
        if sig.dim() == 2:
            sig = sig.unsqueeze(1)
        x = self.stem(sig)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return self.act(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResBlock1D(64, 64, 1), ResBlock1D(64, 64, 1))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, 2), ResBlock1D(128, 128, 1))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, 2), ResBlock1D(256, 256, 1))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, _img, sig):
        if sig.dim() == 2:
            sig = sig.unsqueeze(1)
        x = self.stem(sig)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


class SignalTransformer1D(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.patch = nn.Conv1d(1, d_model, kernel_size=16, stride=16, padding=0)
        tokens = max(1, seq_len // 16)
        self.pos = nn.Parameter(torch.zeros(1, tokens, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, _img, sig):
        if sig.dim() == 2:
            sig = sig.unsqueeze(1)
        x = self.patch(sig).transpose(1, 2)
        pos = self.pos[:, : x.size(1), :]
        x = self.encoder(x + pos)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


class ResNet18Spectrogram(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, img, _sig):
        return self.model(img)


class EfficientNetB0Spectrogram(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        self.model = model

    def forward(self, img, _sig):
        return self.model(img)


class ConvNeXtTinySpectrogram(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        self.model = model

    def forward(self, img, _sig):
        return self.model(img)


def build_model(cfg: dict) -> nn.Module:
    name = str(cfg.get("model", {}).get("name", "signal_resnet1d")).lower()
    num_classes = int(cfg["data"]["num_classes"])
    dropout = float(cfg.get("model", {}).get("dropout", 0.3))
    pretrained = bool(cfg.get("model", {}).get("pretrained", False))
    seq_len = int(cfg["data"].get("target_signal_len", 8192))

    if name == "signal_wdcnn":
        return WDCNN1D(num_classes=num_classes, dropout=dropout)
    if name == "signal_inceptiontime":
        return InceptionTime1D(num_classes=num_classes, dropout=dropout)
    if name == "signal_resnet1d":
        return ResNet1D(num_classes=num_classes, dropout=dropout)
    if name == "signal_transformer":
        return SignalTransformer1D(num_classes=num_classes, seq_len=seq_len)
    if name == "spec_resnet18":
        return ResNet18Spectrogram(num_classes=num_classes, pretrained=pretrained)
    if name == "spec_efficientnet_b0":
        return EfficientNetB0Spectrogram(num_classes=num_classes, pretrained=pretrained)
    if name == "spec_convnext_tiny":
        return ConvNeXtTinySpectrogram(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown model.name: {name}")
