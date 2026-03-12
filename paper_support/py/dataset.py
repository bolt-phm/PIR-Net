
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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def smart_resample(raw_signal, target_len=8192):
    input_len = len(raw_signal)
    if input_len < target_len:
        return cv2.resize(raw_signal.reshape(1, -1), (target_len, 1)).flatten().astype(np.float32)
    factor = input_len // target_len
    if factor < 1: factor = 1
    valid_len = target_len * factor
    cropped = raw_signal[:valid_len]
    reshaped = cropped.reshape(target_len, factor)
    abs_sig = np.abs(reshaped)
    peak_hold = np.max(abs_sig, axis=1)
    energy_trend = np.mean(abs_sig, axis=1)
    downsampled = 0.7 * peak_hold + 0.3 * energy_trend
    return downsampled.astype(np.float32)

def generate_pseudo_image(signal, config):
    if not isinstance(signal, np.ndarray): signal = np.array(signal)
    h, w = tuple(config['data'].get('image_size', [224, 224]))
    nperseg = 256  
    noverlap = int(nperseg * 0.9) 
    channels = []
    try:
        _, _, Zxx = stft(signal, fs=10000, nperseg=nperseg, noverlap=noverlap)
        S_db = 20 * np.log10(np.abs(Zxx) + 1e-6)
        base_img = cv2.resize(normalize_minmax(S_db), (w, h))
        channels.append(base_img)
        grad_f = np.abs(np.gradient(base_img, axis=0))
        grad_t = np.abs(np.gradient(base_img, axis=1))
        channels.append(normalize_minmax(grad_f))
        channels.append(normalize_minmax(grad_t))
        channels.append(normalize_minmax(base_img ** 2))
        time_view = cv2.resize(signal.reshape(1, -1), (w, h))
        channels.append(normalize_minmax(time_view))
    except Exception:
        return np.zeros((h, w, 5), dtype=np.float32)
    return np.stack(channels, axis=-1).astype(np.float32)

def augment_online(img, sig, config):
    aug_cfg = config.get('augment', {})
    sig_cfg = aug_cfg.get('signal', {}).get('params', {})
    img_cfg = aug_cfg.get('image', {})

    # [DEBUG] 打印增强参数，确保参数被正确读取
    # print(f"[DEBUG] Augment params: NoiseP={sig_cfg.get('noise_p')} Amp={sig_cfg.get('noise_amp')}")

    if random.random() < sig_cfg.get('amplitude_scale_p', 0.5):
        scale = random.uniform(*sig_cfg.get('amplitude_scale_range', [0.8, 1.2]))
        sig = sig * scale

    if random.random() < sig_cfg.get('noise_p', 0.5):
        noise_amp = sig_cfg.get('noise_amp', 0.01)
        noise = np.random.randn(len(sig)) * noise_amp * (np.std(sig) + 1e-6)
        sig = sig + noise

    if random.random() < sig_cfg.get('random_cutout_p', 0.0):
        cut_len = int(len(sig) * 0.1)
        start = random.randint(0, len(sig) - cut_len)
        sig[start:start+cut_len] = 0.0

    if random.random() < img_cfg.get('color_jitter_p', 0.0):
        if random.random() < 0.5:
            img = img + random.uniform(-img_cfg.get('brightness', 0.1), img_cfg.get('brightness', 0.1))
        if random.random() < 0.5:
            alpha = random.uniform(1.0 - img_cfg.get('contrast', 0.1), 1.0 + img_cfg.get('contrast', 0.1))
            img = img * alpha
        img = np.clip(img, 0, 1)

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

def normalize_minmax(x):
    mi, ma = np.min(x), np.max(x)
    if ma - mi > 1e-8:
        return (x - mi) / (ma - mi)
    return x.astype(np.float32)

class SmartResampledDataset(Dataset):
    def __init__(self, config, split='train', return_meta=False):
        self.config = config
        self.split = split
        self.is_train = (split == 'train')
        self.return_meta = return_meta
        self.base_resample_factor = 150
        self.target_len_resampled = 8192 
        self.base_raw_len = self.target_len_resampled * self.base_resample_factor
        self.noise_threshold = 0.1 
        self.scales = config['data'].get('scales', [1.0])
        self.indices = self._scan_and_index_segments()
        if self.is_train:
            random.shuffle(self.indices)
        logging.info(f"[{split.upper()}] MultiScale={self.scales}: Indexed {len(self.indices)} segments.")
        
        # [DEBUG] 打印初始化时的增强配置
        aug_force = self.config['augment'].get('force_augment', False)
        logging.info(f"[{split.upper()}] Dataset Init - Force Augment: {aug_force}")

    def _scan_and_index_segments(self):
        indices = []
        data_dir = self.config['data']['data_dir']
        fusion_map = self.config['data'].get('fusion_map', None)
        if fusion_map:
            target_folders = list(fusion_map.keys())
            label_map = fusion_map
        else:
            target_folders = self.config['data']['case_ids']
            label_map = {cid: i for i, cid in enumerate(target_folders)}
        use_whitelist = self.config['data'].get('use_whitelist', False)
        whitelist = set(self.config['data'].get('whitelist', [])) 
        split_ratio = self.config['data'].get('split_ratio', [0.7, 0.15, 0.15])
        overlap_ratio = self.config['data'].get('step_ratio', 0.1) 
        base_step_raw = int(self.base_raw_len * overlap_ratio)
        
        for folder_name in target_folders:
            c_path = os.path.join(data_dir, folder_name)
            if not os.path.exists(c_path): continue
            current_label = label_map[folder_name]
            files = sorted(glob.glob(os.path.join(c_path, "*.npy")))
            for f_path in files:
                if use_whitelist:
                    file_name = os.path.basename(f_path)
                    if file_name not in whitelist: continue
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
                    for scale in self.scales:
                        current_win_len = int(self.base_raw_len * scale)
                        current_step = int(current_win_len * overlap_ratio)
                        if end_idx - start_idx < current_win_len: continue
                        current_ptr = start_idx
                        while current_ptr + current_win_len <= end_idx:
                            indices.append({
                                'path': f_path, 'offset': current_ptr, 'label': current_label,
                                'scale': scale, 'raw_len': current_win_len
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
            raw_file = np.load(meta['path'], mmap_mode='r')
            raw_len = meta['raw_len']
            offset = meta['offset']
            raw_chunk = raw_file[offset : offset + raw_len]
            raw_chunk = np.array(raw_chunk)
            sig = smart_resample(raw_chunk, target_len=self.target_len_resampled)
            global_scale = self.config['data'].get('global_scale', 1.0)
            sig = sig - np.mean(sig)
            sig = sig / (global_scale + 1e-6)
            img = generate_pseudo_image(sig, self.config)
            
            # --- DEBUG: 实时检查配置 ---
            force_augment = self.config['augment'].get('force_augment', False)
            if self.is_train or force_augment:
                # 仅打印一次作为存活证明
                if random.random() < 0.0001: 
                    print(f"✅ [TRACE] Augmenting! ScaleP={self.config['augment']['signal']['params'].get('amplitude_scale_p')} NoiseP={self.config['augment']['signal']['params'].get('noise_p')}")
                img, sig = augment_online(img, sig, self.config)
            
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
            sig_t = torch.tensor(sig).float().unsqueeze(0)
            if self.return_meta:
                meta_info = f"{os.path.basename(meta['path'])}_{meta['offset']}_S{meta['scale']}"
                return img_t, sig_t, meta['label'], meta_info
            return img_t, sig_t, meta['label']
        except Exception:
            return None

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
                    ds, batch_size=config['data']['batch_size'],
                    shuffle=(split == 'train'), 
                    # --- 强制修改：num_workers=0 以便于调试和确保配置传递 ---
                    num_workers=0, 
                    collate_fn=pad_collate, pin_memory=True, persistent_workers=False
                )
            }
        else:
            loaders[split] = {}
    return loaders['train'], loaders['val'], loaders['test']

def run_offline_preprocessing(config): pass
