import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import logging
from scipy.signal import stft
import glob
import warnings

# ----------------------------- #
#          配置与初始化
# ----------------------------- #
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------- #
#       核心信号处理引擎
# ----------------------------- #

def add_awgn(signal, target_snr_db):
    """
    【新增】物理层级的高斯白噪声注入
    根据目标 SNR(dB) 计算噪声功率并叠加。
    公式: P_noise = P_signal / 10^(SNR/10)
    """
    signal = signal.astype(np.float32)
    
    # 计算信号功率 (Mean Square)
    sig_power = np.mean(signal ** 2)
    
    # 静音保护
    if sig_power == 0:
        return signal
    
    # 计算目标噪声功率
    noise_power = sig_power / (10 ** (target_snr_db / 10.0))
    
    # 生成标准正态噪声并缩放
    noise = np.random.randn(*signal.shape).astype(np.float32) * np.sqrt(noise_power)
    
    return signal + noise

def smart_resample(raw_signal, target_len=8192):
    """
    【自适应智能重采样】
    不依赖固定 factor，而是根据输入长度自适应计算。
    确保无论 Scale 是 0.8 还是 1.2，输出永远是 target_len (8192)。
    
    保留: 0.7 * Max(过冲) + 0.3 * Mean(台阶)
    """
    input_len = len(raw_signal)
    
    # 如果数据不够长，先强制线性插值拉伸 (这种情况极少见，但需容错)
    if input_len < target_len:
        return cv2.resize(raw_signal.reshape(1, -1), (target_len, 1)).flatten().astype(np.float32)
        
    # 计算自适应降采样因子
    factor = input_len // target_len
    if factor < 1: factor = 1
    
    # 为了保证整除，截断多余的尾部
    valid_len = target_len * factor
    cropped = raw_signal[:valid_len]
    
    # Reshape 进行池化
    reshaped = cropped.reshape(target_len, factor)
    abs_sig = np.abs(reshaped)
    
    # 混合池化策略
    peak_hold = np.max(abs_sig, axis=1)
    energy_trend = np.mean(abs_sig, axis=1)
    
    downsampled = 0.7 * peak_hold + 0.3 * energy_trend
    return downsampled.astype(np.float32)

def normalize_robust(x):
    """ 
    【新增】鲁棒归一化 
    先进行 3-sigma 截断，去除极端噪声尖峰，再做 MinMax。
    防止强噪声（如 0dB）下的个别极值导致整体图像变黑。
    """
    mu = np.mean(x)
    std = np.std(x)
    
    # 截断区间 [mu - 3std, mu + 3std]
    lower, upper = mu - 3 * std, mu + 3 * std
    x = np.clip(x, lower, upper)
    
    # 标准 MinMax
    mi, ma = np.min(x), np.max(x)
    if ma - mi > 1e-8:
        return (x - mi) / (ma - mi)
    return x.astype(np.float32)

def normalize_minmax(x):
    """ 传统归一化 (保留用于非敏感数据) """
    mi, ma = np.min(x), np.max(x)
    if ma - mi > 1e-8:
        return (x - mi) / (ma - mi)
    return x.astype(np.float32)

def generate_pseudo_image(signal, config):
    """ 生成 5 通道高分辨率 STFT 特征图 """
    if not isinstance(signal, np.ndarray): signal = np.array(signal)
    h, w = tuple(config['data'].get('image_size', [224, 224]))
    
    # 高分辨率配置 (nperseg=256)
    nperseg = 256  
    noverlap = int(nperseg * 0.9) 
    
    channels = []
    try:
        # 1. STFT 幅度谱
        _, _, Zxx = stft(signal, fs=10000, nperseg=nperseg, noverlap=noverlap)
        S_db = 20 * np.log10(np.abs(Zxx) + 1e-6)
        # 使用鲁棒归一化
        base_img = cv2.resize(normalize_robust(S_db), (w, h))
        channels.append(base_img)
        
        # 2. 梯度特征 (捕捉边缘)
        grad_f = np.abs(np.gradient(base_img, axis=0))
        grad_t = np.abs(np.gradient(base_img, axis=1))
        channels.append(normalize_robust(grad_f))
        channels.append(normalize_robust(grad_t))
        
        # 3. 能量特征
        channels.append(normalize_robust(base_img ** 2))
        
        # 4. 时域波形可视化
        time_view = cv2.resize(signal.reshape(1, -1), (w, h))
        channels.append(normalize_robust(time_view))
        
    except Exception:
        # 容错：全黑图
        return np.zeros((h, w, 5), dtype=np.float32)

    return np.stack(channels, axis=-1).astype(np.float32)

def augment_online(img, sig, config):
    """ 
    训练时数据增强 
    【修改】移除了旧的 noise 添加逻辑，统一在 __getitem__ 中处理 SNR 噪声
    """
    aug_cfg = config.get('augment', {})
    sig_cfg = aug_cfg.get('signal', {}).get('params', {})
    img_cfg = aug_cfg.get('image', {})

    # 1. 信号幅度缩放 (不改变 SNR，只改变增益)
    if random.random() < sig_cfg.get('amplitude_scale_p', 0.5):
        scale = random.uniform(*sig_cfg.get('amplitude_scale_range', [0.8, 1.2]))
        sig = sig * scale

    # 2. 信号随机遮挡 (模拟丢包)
    if random.random() < sig_cfg.get('random_cutout_p', 0.0):
        cut_len = int(len(sig) * 0.1)
        if len(sig) > cut_len:
            start = random.randint(0, len(sig) - cut_len)
            sig[start:start+cut_len] = 0.0

    # 3. 图像颜色抖动 (亮度/对比度)
    if random.random() < img_cfg.get('color_jitter_p', 0.0):
        if random.random() < 0.5:
            # 亮度
            img = img + random.uniform(-img_cfg.get('brightness', 0.1), img_cfg.get('brightness', 0.1))
        if random.random() < 0.5:
            # 对比度
            alpha = random.uniform(1.0 - img_cfg.get('contrast', 0.1), 1.0 + img_cfg.get('contrast', 0.1))
            img = img * alpha
        img = np.clip(img, 0, 1)

    # 4. 图像随机擦除 (模拟局部频谱丢失)
    if random.random() < img_cfg.get('random_erasing_p', 0.0):
        h, w, c = img.shape
        area_ratio = random.uniform(*img_cfg.get('erasing_area_range', [0.02, 0.1]))
        target_area = h * w * area_ratio
        aspect_ratio = random.uniform(*img_cfg.get('erasing_ratio_range', [0.3, 3.3]))
        
        eh = int(round(np.sqrt(target_area * aspect_ratio)))
        ew = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if eh < h and ew < w:
            y = random.randint(0, h - eh)
            x = random.randint(0, w - ew)
            img[y:y+eh, x:x+ew, :] = 0.0

    return img.astype(np.float32), sig.astype(np.float32)


# ----------------------------- #
#       数据集核心类
# ----------------------------- #

class SmartResampledDataset(Dataset):
    def __init__(self, config, split='train', return_meta=False):
        self.config = config
        self.split = split
        self.is_train = (split == 'train')
        self.return_meta = return_meta
        
        # 基础参数
        # 1.0 尺度下的原始长度 = 8192 * 150 = 1,228,800
        self.base_resample_factor = 150
        self.target_len_resampled = 8192 
        self.base_raw_len = self.target_len_resampled * self.base_resample_factor
        
        self.noise_threshold = 0.1 
        
        # 多尺度配置: 默认为 [1.0]
        self.scales = config['data'].get('scales', [1.0])
        
        # 扫描数据（含白名单过滤逻辑）
        self.indices = self._scan_and_index_segments()
        
        if self.is_train:
            random.shuffle(self.indices)
            
        logging.info(f"[{split.upper()}] MultiScale={self.scales}: Indexed {len(self.indices)} segments.")

    def _scan_and_index_segments(self):
        """ 
        支持 Fusion Map + Multi-Scale + Time Splitting + 【白名单过滤】
        """
        indices = []
        data_dir = self.config['data']['data_dir']
        
        # 1. 路径与标签解析
        fusion_map = self.config['data'].get('fusion_map', None)
        if fusion_map:
            target_folders = list(fusion_map.keys())
            label_map = fusion_map
        else:
            target_folders = self.config['data']['case_ids']
            label_map = {cid: i for i, cid in enumerate(target_folders)}

        # 【新增】白名单逻辑初始化
        use_whitelist = self.config['data'].get('use_whitelist', False)
        whitelist = set(self.config['data'].get('whitelist', [])) 

        # 2. 切分参数
        split_ratio = self.config['data'].get('split_ratio', [0.7, 0.15, 0.15])
        overlap_ratio = self.config['data'].get('step_ratio', 0.1) 
        
        # 基础步长 (1.0尺度下)
        base_step_raw = int(self.base_raw_len * overlap_ratio)
        
        for folder_name in target_folders:
            c_path = os.path.join(data_dir, folder_name)
            if not os.path.exists(c_path): continue
            
            current_label = label_map[folder_name]
            files = sorted(glob.glob(os.path.join(c_path, "*.npy")))
            
            for f_path in files:
                # 白名单过滤
                if use_whitelist:
                    file_name = os.path.basename(f_path)
                    if file_name not in whitelist:
                        continue

                try:
                    raw_file_ref = np.load(f_path, mmap_mode='r')
                    total_len = len(raw_file_ref)
                    
                    train_end = int(total_len * split_ratio[0])
                    val_end = int(total_len * (split_ratio[0] + split_ratio[1]))
                    
                    if self.split == 'train':
                        start_idx, end_idx = 0, train_end
                    elif self.split == 'val':
                        start_idx, end_idx = train_end, val_end
                    else:
                        start_idx, end_idx = val_end, total_len
                    
                    # 多尺度循环
                    for scale in self.scales:
                        current_win_len = int(self.base_raw_len * scale)
                        current_step = int(current_win_len * overlap_ratio)
                        
                        if end_idx - start_idx < current_win_len: continue
                        
                        current_ptr = start_idx
                        while current_ptr + current_win_len <= end_idx:
                            indices.append({
                                'path': f_path,
                                'offset': current_ptr,
                                'label': current_label,
                                'scale': scale,          
                                'raw_len': current_win_len
                            })
                            current_ptr += current_step
                        
                except Exception as e:
                    logging.warning(f"Error scanning {f_path}: {e}")
                    continue
                
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        meta = self.indices[idx]
        try:
            # 1. 精确读取原始数据
            raw_file = np.load(meta['path'], mmap_mode='r')
            raw_len = meta['raw_len']
            offset = meta['offset']
            
            raw_chunk = raw_file[offset : offset + raw_len]
            raw_chunk = np.array(raw_chunk)
            
            # 2. 自适应智能重采样 -> 8192 点
            sig = smart_resample(raw_chunk, target_len=self.target_len_resampled)
            
            # 3. 全局归一化
            global_scale = self.config['data'].get('global_scale', 1.0)
            sig = sig - np.mean(sig)
            sig = sig / (global_scale + 1e-6)
            
            # ========================================================
            # 🚀 核心修改：在生成图像之前，从源头注入物理噪声 (SNR)
            # ========================================================
            
            # 读取配置
            aug_cfg = self.config.get('augment', {}).get('signal', {})
            aug_params = aug_cfg.get('params', {})
            use_aug = aug_cfg.get('use_augment', False)

            # A. 训练模式：随机注入噪声 (Robust Training)
            if self.is_train and use_aug:
                # 检查概率
                if random.random() < aug_params.get('noise_p', 0.0):
                    # 获取分贝范围
                    min_snr = aug_params.get('min_snr_db', -5.0)
                    max_snr = aug_params.get('max_snr_db', 25.0)
                    # 随机选择 SNR
                    target_snr = random.uniform(min_snr, max_snr)
                    sig = add_awgn(sig, target_snr)

            # B. 非训练模式 (Val/Test)：检查是否有强制测试指令
            elif not self.is_train:
                forced_snr_str = os.environ.get('FORCE_SNR')
                if forced_snr_str is not None and forced_snr_str.strip() != "":
                    try:
                        snr_val = float(forced_snr_str)
                        sig = add_awgn(sig, snr_val)
                    except ValueError:
                        pass
            
            # ========================================================

            # 4. 生成图像
            # 此时 sig 已带噪，STFT 会自动将噪声转换为频域特征
            # generate_pseudo_image 内部已更新为 normalize_robust
            img = generate_pseudo_image(sig, self.config)
            
            # 5. 其他增强 (训练时叠加 Cutout, ColorJitter 等)
            if self.is_train:
                img, sig = augment_online(img, sig, self.config)
            
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
            sig_t = torch.tensor(sig).float().unsqueeze(0)
            
            if self.return_meta:
                meta_info = f"{os.path.basename(meta['path'])}_{meta['offset']}_S{meta['scale']}"
                return img_t, sig_t, meta['label'], meta_info
            
            return img_t, sig_t, meta['label']
            
        except Exception as e:
            # 调试时可解开下面注释
            # logging.error(f"Error loading {meta['path']}: {e}")
            return None

# ----------------------------- #
#       DataLoader 工厂
# ----------------------------- #

def pad_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    elem_len = len(batch[0])
    imgs = torch.stack([x[0] for x in batch])
    sigs = torch.stack([x[1] for x in batch])
    lbls = torch.tensor([x[2] for x in batch], dtype=torch.long)
    
    if elem_len == 4:
        metas = [x[3] for x in batch]
        return imgs, sigs, lbls, metas
    
    return imgs, sigs, lbls

def create_dataloaders(config, return_meta=False):
    loaders = {}
    loader_key = "SmartResample_F150"
    
    for split in ['train', 'val', 'test']:
        ds = SmartResampledDataset(config, split=split, return_meta=return_meta)
        
        if len(ds) > 0:
            loaders[split] = {
                loader_key: DataLoader(
                    ds,
                    batch_size=config['data']['batch_size'],
                    shuffle=(split == 'train'), 
                    num_workers=config['data']['num_workers'],
                    collate_fn=pad_collate,
                    pin_memory=True,
                    persistent_workers=True
                )
            }
        else:
            loaders[split] = {}
            
    return loaders['train'], loaders['val'], loaders['test']

def run_offline_preprocessing(config): pass