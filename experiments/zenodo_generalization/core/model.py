import torch
import torch.nn as nn


class WDCNN1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, _img, sig):
        x = self.features(sig)
        return self.head(x)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        b = out_ch // 4
        self.b1 = nn.Conv1d(in_ch, b, kernel_size=1, padding=0, bias=False)
        self.b3 = nn.Conv1d(in_ch, b, kernel_size=3, padding=1, bias=False)
        self.b5 = nn.Conv1d(in_ch, b, kernel_size=5, padding=2, bias=False)
        self.b7 = nn.Conv1d(in_ch, b, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.b7(x)], dim=1)
        return self.act(self.bn(x))


class InceptionTime1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        blocks = []
        ch = 64
        for _ in range(6):
            blocks.append(InceptionBlock1D(ch, 128))
            ch = 128
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, _img, sig):
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
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return self.act(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
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
        x = self.stem(sig)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


class SignalTransformer1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch = nn.Conv1d(in_channels, d_model, kernel_size=16, stride=16, padding=0)
        tokens = max(1, seq_len // 16)
        self.pos = nn.Parameter(torch.zeros(1, tokens, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, _img, sig):
        x = self.patch(sig).transpose(1, 2)
        pos = self.pos[:, : x.size(1), :]
        x = self.encoder(x + pos)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


class CNNBiLSTMAttn1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.rnn = nn.LSTM(
            input_size=192,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.attn = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, _img, sig):
        x = self.frontend(sig)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        w = self.attn(x).squeeze(-1)
        w = torch.softmax(w, dim=1).unsqueeze(-1)
        x = torch.sum(x * w, dim=1)
        return self.head(x)


class PIRImageEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.proj(x)


class PIRSignalEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, backbone: str = "hybrid", dropout: float = 0.2):
        super().__init__()
        self.backbone = str(backbone).lower()
        hidden = max(64, embed_dim)
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        tr_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, min(8, hidden // 32)),
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(tr_layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, embed_dim)

    def forward(self, sig):
        x = self.frontend(sig)
        if self.backbone == "cnn":
            pooled = x.mean(dim=-1)
            return self.proj(pooled)

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        if self.backbone == "hybrid":
            x = self.transformer(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.proj(pooled)


class PIRNetDualBranch(nn.Module):
    def __init__(
        self,
        sig_in_channels: int,
        img_in_channels: int,
        num_classes: int,
        embed_dim: int = 192,
        dropout: float = 0.3,
        fusion_mode: str = "cross_attention",
        branch_mode: str = "dual",
        signal_backbone: str = "hybrid",
        attn_heads: int = 4,
    ):
        super().__init__()
        self.branch_mode = str(branch_mode).lower()
        self.fusion_mode = str(fusion_mode).lower()

        self.use_image = self.branch_mode in {"dual", "image_only"}
        self.use_signal = self.branch_mode in {"dual", "signal_only"}
        if not (self.use_image or self.use_signal):
            raise ValueError(f"Invalid branch_mode={self.branch_mode}")

        if self.use_image:
            self.image_encoder = PIRImageEncoder(in_channels=img_in_channels, embed_dim=embed_dim)
        if self.use_signal:
            self.signal_encoder = PIRSignalEncoder(
                in_channels=sig_in_channels,
                embed_dim=embed_dim,
                backbone=signal_backbone,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(dropout)
        if self.use_image and self.use_signal and self.fusion_mode == "cross_attention":
            self.fusion_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=max(1, int(attn_heads)), batch_first=True)
            self.fusion_norm = nn.LayerNorm(embed_dim)
        elif self.use_image and self.use_signal and self.fusion_mode == "simple_attention":
            self.fusion_gate = nn.Linear(embed_dim * 2, 2)

        if self.use_image and self.use_signal and self.fusion_mode == "concat":
            cls_in = embed_dim * 2
            self.concat_proj = nn.Sequential(
                nn.Linear(cls_in, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
            )
            cls_in = embed_dim
        else:
            cls_in = embed_dim
        self.classifier = nn.Linear(cls_in, num_classes)

    def _fuse(self, img_emb: torch.Tensor | None, sig_emb: torch.Tensor | None) -> torch.Tensor:
        if img_emb is None and sig_emb is None:
            raise RuntimeError("Both embeddings are None.")
        if img_emb is None:
            return sig_emb
        if sig_emb is None:
            return img_emb

        if self.fusion_mode == "cross_attention":
            tokens = torch.stack([img_emb, sig_emb], dim=1)
            out, _ = self.fusion_attn(tokens, tokens, tokens, need_weights=False)
            return self.fusion_norm(out.mean(dim=1))
        if self.fusion_mode == "simple_attention":
            gate = torch.softmax(self.fusion_gate(torch.cat([img_emb, sig_emb], dim=-1)), dim=-1)
            return gate[:, 0:1] * img_emb + gate[:, 1:2] * sig_emb
        if self.fusion_mode == "concat":
            return self.concat_proj(torch.cat([img_emb, sig_emb], dim=-1))
        return 0.5 * (img_emb + sig_emb)

    def forward(self, img, sig):
        img_emb = self.image_encoder(img) if self.use_image else None
        sig_emb = self.signal_encoder(sig) if self.use_signal else None
        fused = self._fuse(img_emb, sig_emb)
        fused = self.dropout(fused)
        return self.classifier(fused)


def build_model(cfg: dict) -> nn.Module:
    name = str(cfg.get("model", {}).get("name", "signal_resnet1d")).lower()
    num_classes = int(cfg["data"]["num_classes"])
    dropout = float(cfg.get("model", {}).get("dropout", 0.3))
    seq_len = int(round(float(cfg["data"].get("window_seconds", 4.0)) * float(cfg["data"].get("sample_rate", 500))))
    in_channels = int(cfg.get("model", {}).get("in_channels", len(cfg["data"].get("channels", ["A", "B", "C", "D", "E", "F", "Gx", "Gy"]))))

    if name == "signal_wdcnn":
        return WDCNN1D(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    if name == "signal_inceptiontime":
        return InceptionTime1D(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    if name == "signal_resnet1d":
        return ResNet1D(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    if name == "signal_transformer":
        return SignalTransformer1D(in_channels=in_channels, num_classes=num_classes, seq_len=seq_len)
    if name == "signal_cnn_bilstm_attn":
        return CNNBiLSTMAttn1D(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    if name in {"pirnet_dual_branch", "pirnet", "pirnet_lite"}:
        image_in_channels = int(cfg["data"].get("image_out_channels", 3))
        embed_dim = int(cfg.get("model", {}).get("embed_dim", 192))
        fusion_mode = str(cfg.get("model", {}).get("fusion_mode", "cross_attention"))
        branch_mode = str(cfg.get("model", {}).get("branch_mode", "dual"))
        signal_backbone = str(cfg.get("model", {}).get("signal_backbone", "hybrid"))
        attn_heads = int(cfg.get("model", {}).get("attn_heads", 4))
        return PIRNetDualBranch(
            sig_in_channels=in_channels,
            img_in_channels=image_in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=dropout,
            fusion_mode=fusion_mode,
            branch_mode=branch_mode,
            signal_backbone=signal_backbone,
            attn_heads=attn_heads,
        )

    raise ValueError(f"Unknown model.name: {name}")

