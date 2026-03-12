import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 1. 基础组件 (Basic Blocks)
# ==============================================================================

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Block (通道注意力) """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScaleConvBlock(nn.Module):
    """ 多尺度卷积块 (Inception-like): 1x1, 3x3, 5x5, 7x7 """
    def __init__(self, in_channels, out_channels, stride=1):
        super(MultiScaleConvBlock, self).__init__()
        
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(branch_channels), nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels), nn.ReLU(inplace=True),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(branch_channels), nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels), nn.ReLU(inplace=True),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm1d(branch_channels), nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels), nn.ReLU(inplace=True),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm1d(branch_channels), nn.ReLU(inplace=True)
        )
        
        self.final_channels = branch_channels * 4
        self.adjust_conv = None
        if self.final_channels != out_channels:
             self.adjust_conv = nn.Conv1d(self.final_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        if self.adjust_conv is not None:
            out = self.adjust_conv(out)
        return out


class PositionalEncoding(nn.Module):
    """ Transformer 位置编码 """
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# 2. 图像编码器 (Image Encoder)
# ==============================================================================

class SimpleCNN(nn.Module):
    """ 【Type 0】轻量级简单 CNN (找回的功能) """
    def __init__(self, in_channels, out_dim):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> 28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # -> 1x1
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

def get_image_encoder(config):
    model_cfg = config['modality']['image_model']
    encoder_type = model_cfg['type']
    out_dim = model_cfg['out_dim']
    # 自动适配 5 通道
    in_channels = 5 if config['modality'].get('pseudo_image_mode', 0) == 1 else 1
    pretrained = model_cfg.get('pretrained', True)

    # Type 0: 简单 CNN
    if encoder_type == 0:
        return SimpleCNN(in_channels, out_dim)

    # Type 1: ResNet50, Type 2: ResNet101
    elif encoder_type in [1, 2]: 
        weights = (ResNet50_Weights.IMAGENET1K_V2 if encoder_type == 1 else ResNet101_Weights.IMAGENET1K_V2) if pretrained else None
        model = resnet50(weights=weights) if encoder_type == 1 else resnet101(weights=weights)
        
        # 修改第一层适配多通道
        original_conv1 = model.conv1
        new_conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=False
        )
        
        if pretrained:
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                mean_weight = torch.mean(original_conv1.weight, dim=1, keepdim=True)
                for i in range(3, in_channels):
                    new_conv1.weight[:, i:i+1, :, :] = mean_weight
        
        model.conv1 = new_conv1
        model.fc = nn.Linear(model.fc.in_features, out_dim)
        return model
    else:
        raise ValueError(f"Invalid image encoder type: {encoder_type}")


# ==============================================================================
# 3. 信号编码器 (Signal Encoder)
# ==============================================================================

class SimpleRNN(nn.Module):
    """ 【Type 0】修复版 RNN: 处理维度不匹配问题 """
    def __init__(self, inp, outp):
        super().__init__()
        # inp 通常为 1
        self.rnn = nn.RNN(inp, 128, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(128, outp) 

    def forward(self, x):
        # x shape: (Batch, 1, 8192) or (Batch, 8192, 1)
        
        # 1. 维度修正: 确保变为 (Batch, Seq_Len, Features) -> (B, 8192, 1)
        if x.dim() == 3 and x.size(1) == 1: 
            x = x.permute(0, 2, 1) 
        elif x.dim() == 2: 
            x = x.unsqueeze(2) 

        # 2. RNN 前向
        out, _ = self.rnn(x) # -> (Batch, 8192, 128)
        
        # 3. 取最后一个时间步
        out = out[:, -1, :]  # -> (Batch, 128)
        
        # 4. 映射输出
        return self.fc(out)


class SimpleLSTM(nn.Module):
    """ 【Type 1】LSTM 封装类 """
    def __init__(self, inp, outp):
        super().__init__()
        self.lstm = nn.LSTM(inp, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, outp) # 双向 128*2
        
    def forward(self, x):
        # 维度修正
        if x.dim() == 3 and x.size(1) == 1: 
            x = x.permute(0, 2, 1)
        elif x.dim() == 2: 
            x = x.unsqueeze(2)
            
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # 取最后一帧


class SignalHybridEncoder(nn.Module):
    """ 【Type 2】混合 SOTA 编码器 (CNN+LSTM+Transformer) """
    def __init__(self, in_channels, out_dim, embed_dim, nhead, num_layers, dropout):
        super(SignalHybridEncoder, self).__init__()
        
        # CNN Stem
        self.cnn_stem = nn.Sequential(
            MultiScaleConvBlock(in_channels, 32, stride=2), SEBlock(32),
            MultiScaleConvBlock(32, 64, stride=2), SEBlock(64),
            MultiScaleConvBlock(64, 64, stride=2), SEBlock(64),
            MultiScaleConvBlock(64, 128, stride=2), SEBlock(128),
            MultiScaleConvBlock(128, 128, stride=2), SEBlock(128),
            MultiScaleConvBlock(128, 256, stride=2), SEBlock(256)
        )

        self.projection = nn.Linear(256, embed_dim * 2) 
        self.lstm = nn.LSTM(embed_dim * 2, embed_dim, batch_first=True, bidirectional=True)
        
        self.pos_encoder = PositionalEncoding(embed_dim * 2, dropout, max_len=5000) 
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * 2, nhead=nhead, dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 1, 8192)
        x = self.cnn_stem(x) # -> (B, 256, 128)
        x = x.permute(0, 2, 1) # -> (B, 128, 256)
        x = self.projection(x) 
        
        self.lstm.flatten_parameters() 
        x, _ = self.lstm(x)    # -> (B, 128, embed*2)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1) # GAP
        x = self.dropout(x)
        x = self.fc(x)
        return x

def get_signal_encoder(config):
    model_cfg = config['modality']['signal_model']
    encoder_type = model_cfg['type']
    in_channels = model_cfg['in_channels']
    out_dim = config['modality']['image_model']['out_dim']

    if encoder_type == 0:
        return SimpleRNN(in_channels, out_dim)
    elif encoder_type == 1:
        return SimpleLSTM(in_channels, out_dim)
    elif encoder_type == 2: 
        embed_dim = model_cfg.get('embed_dim', 256)
        nhead = model_cfg.get('nhead', 8)
        num_layers = model_cfg.get('num_layers', 2)
        dropout = model_cfg.get('dropout', 0.5)
        return SignalHybridEncoder(in_channels, out_dim, embed_dim, nhead, num_layers, dropout)
    else: 
        raise ValueError(f"Invalid signal encoder type: {encoder_type}")


# ==============================================================================
# 4. 融合模块 (Fusion)
# ==============================================================================

class AttentionFusion(nn.Module):
    """ 【Type 1】简单注意力融合 """
    def __init__(self, input_dim, dropout=0.0):
        super(AttentionFusion, self).__init__()
        self.W_i = nn.Linear(input_dim, input_dim)
        self.W_s = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img, sig):
        alpha_i = self.v(torch.tanh(self.W_i(img)))
        alpha_s = self.v(torch.tanh(self.W_s(sig)))
        weights = torch.softmax(torch.cat([alpha_i, alpha_s], dim=1), dim=1)
        out = weights[:, 0].unsqueeze(1) * img + weights[:, 1].unsqueeze(1) * sig
        return self.dropout(out)

class MultiHeadAttentionFusion(nn.Module):
    """ 【Type 2】多头注意力融合 """
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img, sig):
        query = img.unsqueeze(1)
        key = torch.stack([img, sig], dim=1)
        value = key
        
        attn_out, _ = self.multihead_attn(query=query, key=key, value=value)
        fused = attn_out.squeeze(1) + img 
        return self.dropout(self.layer_norm(fused))

def get_fusion_module(config):
    fusion_cfg = config['modality']['fusion']
    fusion_type = fusion_cfg['type']
    input_dim = config['modality']['image_model']['out_dim']
    dropout_val = fusion_cfg.get('dropout', 0.5)
    
    if fusion_type == 1: 
        return AttentionFusion(input_dim, dropout=dropout_val)
    elif fusion_type == 2: 
        return MultiHeadAttentionFusion(input_dim, fusion_cfg.get('num_heads', 4), dropout=dropout_val)
    else: 
        # Type 0 (default): Concat + Linear
        return nn.Linear(input_dim*2, input_dim)


# ==============================================================================
# 5. 主模型 (Main Model)
# ==============================================================================

class MultiModalNet(nn.Module):
    def __init__(self, config):
        super(MultiModalNet, self).__init__()
        self.image_encoder = get_image_encoder(config)
        self.signal_encoder = get_signal_encoder(config)
        self.fusion = get_fusion_module(config)
        self.config = config

        fusion_out_dim = config['modality']['image_model']['out_dim']
        global_dropout = config['modality']['signal_model'].get('dropout', 0.5)
        num_classes = config['data']['num_classes']
        
        self.classifier = nn.Sequential(
            nn.Dropout(global_dropout),
            nn.Linear(fusion_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(global_dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, image_data, signal_data):
        # 信号维度统一修正为 (B, 1, L) 方便传递给 Hybrid Encoder
        # SimpleRNN/LSTM 内部会自己再处理一遍转置
        if signal_data.dim() == 2: 
            signal_data = signal_data.unsqueeze(1)
        
        image_features = self.image_encoder(image_data)
        signal_features = self.signal_encoder(signal_data)
        
        # 融合逻辑
        fusion_type = self.config['modality']['fusion']['type']
        
        if fusion_type in [1, 2]: # Attention 类
            fused_features = self.fusion(image_features, signal_features)
        else: # Type 0: Concat
            combined = torch.cat([image_features, signal_features], dim=1)
            fused_features = self.fusion(combined) 
            
        output = self.classifier(fused_features)
        return output

def build_model(config):
    return MultiModalNet(config)