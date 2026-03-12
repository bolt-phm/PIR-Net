import os
import glob
import logging
import random
import warnings

import cv2
import numpy as np
import torch
from scipy.signal import stft
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def add_awgn_numpy(signal, snr_db):
    if snr_db is None:
        return signal
    signal = signal.astype(np.float32)
    signal_power = np.mean(signal ** 2)
    if signal_power == 0:
        return signal
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(*signal.shape).astype(np.float32)
    return signal + noise * np.sqrt(noise_power)


def smart_resample(raw_signal, target_len=8192, alpha=0.7):
    input_len = len(raw_signal)
    if input_len < target_len:
        return cv2.resize(raw_signal.reshape(1, -1), (target_len, 1)).flatten().astype(np.float32)

    factor = max(1, input_len // target_len)
    valid_len = target_len * factor
    cropped = raw_signal[:valid_len]
    reshaped = cropped.reshape(target_len, factor)
    abs_sig = np.abs(reshaped)

    peak_hold = np.max(abs_sig, axis=1)
    energy_trend = np.mean(abs_sig, axis=1)

    downsampled = alpha * peak_hold + (1.0 - alpha) * energy_trend
    return downsampled.astype(np.float32)


def normalize_minmax(x):
    mi, ma = np.min(x), np.max(x)
    if ma - mi > 1e-8:
        return (x - mi) / (ma - mi)
    return x.astype(np.float32)


def generate_pseudo_image(signal, config):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    h, w = tuple(config['data'].get('image_size', [224, 224]))
    nperseg = config['data'].get('stft_nperseg', 256)
    noverlap = int(nperseg * config['data'].get('stft_overlap_ratio', 0.9))

    channels = []
    try:
        _, _, zxx = stft(signal, fs=10000, nperseg=nperseg, noverlap=noverlap)
        s_db = 20 * np.log10(np.abs(zxx) + 1e-6)
        base_img = cv2.resize(normalize_minmax(s_db), (w, h))
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
        sig[start:start + cut_len] = 0.0

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
            img[y:y + eh, x:x + ew, :] = 0.0

    return img.astype(np.float32), sig.astype(np.float32)


class SmartResampledDataset(Dataset):
    def __init__(self, config, split='train', return_meta=False):
        self.config = config
        self.split = split
        self.is_train = split == 'train'
        self.return_meta = return_meta

        self.base_resample_factor = int(config['data'].get('base_resample_factor', 150))
        self.target_len_resampled = int(config['data'].get('target_signal_len', 8192))
        self.base_raw_len = self.target_len_resampled * self.base_resample_factor
        self.scales = config['data'].get('scales', [1.0])
        self.split_ratio = config['data'].get('split_ratio', [0.7, 0.15, 0.15])
        self.overlap_ratio = config['data'].get('step_ratio', 0.1)
        self.split_mode = config['data'].get('split_mode', 'temporal').lower()
        self.file_split_seed = int(config['data'].get('file_split_seed', config['train'].get('seed', 42)))

        self.indices = self._scan_and_index_segments()
        if self.is_train:
            random.shuffle(self.indices)

        logging.info(f"[{split.upper()}] mode={self.split_mode}, scales={self.scales}, samples={len(self.indices)}")

    def _resolve_targets(self):
        fusion_map = self.config['data'].get('fusion_map', None)
        if fusion_map:
            target_folders = list(fusion_map.keys())
            label_map = fusion_map
        else:
            target_folders = self.config['data']['case_ids']
            label_map = {cid: i for i, cid in enumerate(target_folders)}
        return target_folders, label_map

    def _iter_valid_files(self, folder_name):
        data_dir = self.config['data']['data_dir']
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path):
            return []

        use_whitelist = self.config['data'].get('use_whitelist', False)
        whitelist = set(self.config['data'].get('whitelist', []))

        files = sorted(glob.glob(os.path.join(folder_path, '*.npy')))
        if not use_whitelist:
            return files
        return [fp for fp in files if os.path.basename(fp) in whitelist]

    @staticmethod
    def _safe_split_counts(n, ratios):
        if n <= 0:
            return 0, 0, 0
        r_train, r_val, r_test = ratios
        train_n = int(n * r_train)
        val_n = int(n * r_val)
        test_n = n - train_n - val_n

        if n >= 3:
            if train_n == 0:
                train_n = 1
            if val_n == 0:
                val_n = 1
            test_n = n - train_n - val_n
            if test_n <= 0:
                if train_n > val_n:
                    train_n -= 1
                else:
                    val_n -= 1
                test_n = n - train_n - val_n
        return train_n, val_n, test_n

    def _file_split(self, files):
        files = list(files)
        rng = random.Random(self.file_split_seed)
        rng.shuffle(files)

        train_n, val_n, _ = self._safe_split_counts(len(files), self.split_ratio)
        train_files = set(files[:train_n])
        val_files = set(files[train_n:train_n + val_n])
        test_files = set(files[train_n + val_n:])

        if self.split == 'train':
            return train_files
        if self.split == 'val':
            return val_files
        return test_files

    def _window_ranges_temporal(self, total_len):
        train_end = int(total_len * self.split_ratio[0])
        val_end = int(total_len * (self.split_ratio[0] + self.split_ratio[1]))
        if self.split == 'train':
            return 0, train_end
        if self.split == 'val':
            return train_end, val_end
        return val_end, total_len

    def _window_ranges_file_mode(self, total_len):
        return 0, total_len

    def _scan_and_index_segments(self):
        indices = []
        target_folders, label_map = self._resolve_targets()

        for folder_name in target_folders:
            files = self._iter_valid_files(folder_name)
            if not files:
                continue

            if self.split_mode == 'file':
                active_files = self._file_split(files)
            else:
                active_files = set(files)

            label = int(label_map[folder_name])
            for file_path in files:
                if file_path not in active_files:
                    continue
                try:
                    raw_ref = np.load(file_path, mmap_mode='r')
                    total_len = len(raw_ref)
                except Exception as e:
                    logging.warning(f"Skip bad file {file_path}: {e}")
                    continue

                if self.split_mode == 'file':
                    start_idx, end_idx = self._window_ranges_file_mode(total_len)
                else:
                    start_idx, end_idx = self._window_ranges_temporal(total_len)

                for scale in self.scales:
                    win_len = int(self.base_raw_len * scale)
                    step = max(1, int(win_len * self.overlap_ratio))
                    if end_idx - start_idx < win_len:
                        continue

                    ptr = start_idx
                    while ptr + win_len <= end_idx:
                        indices.append({
                            'path': file_path,
                            'offset': ptr,
                            'label': label,
                            'scale': scale,
                            'raw_len': win_len,
                        })
                        ptr += step

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        meta = self.indices[idx]
        try:
            raw_file = np.load(meta['path'], mmap_mode='r')
            raw_chunk = np.array(raw_file[meta['offset']: meta['offset'] + meta['raw_len']])

            alpha = float(self.config['data'].get('resample_alpha', 0.7))
            sig = smart_resample(raw_chunk, target_len=self.target_len_resampled, alpha=alpha)

            global_scale = float(self.config['data'].get('global_scale', 1.0))
            sig = sig - np.mean(sig)
            sig = sig / (global_scale + 1e-6)

            forced_snr_str = os.environ.get('FORCE_SNR')
            if forced_snr_str is not None and forced_snr_str.strip() != '':
                try:
                    sig = add_awgn_numpy(sig, float(forced_snr_str))
                except ValueError:
                    pass

            img = generate_pseudo_image(sig, self.config)
            if self.is_train:
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
    if not batch:
        return None

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
    loader_key = 'SmartResample_F150'
    num_workers = int(config['data'].get('num_workers', 0))
    pin_memory = bool(torch.cuda.is_available())

    for split in ['train', 'val', 'test']:
        ds = SmartResampledDataset(config, split=split, return_meta=return_meta)
        if len(ds) > 0:
            loaders[split] = {
                loader_key: DataLoader(
                    ds,
                    batch_size=config['data']['batch_size'],
                    shuffle=(split == 'train'),
                    num_workers=num_workers,
                    collate_fn=pad_collate,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
            }
        else:
            loaders[split] = {}

    return loaders['train'], loaders['val'], loaders['test']


def run_offline_preprocessing(config):
    return None
