import os
import glob
import random
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.signal import stft
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def add_awgn_numpy(signal: np.ndarray, snr_db: float | None) -> np.ndarray:
    if snr_db is None:
        return signal
    signal = signal.astype(np.float32)
    power = float(np.mean(signal ** 2))
    if power <= 1e-12:
        return signal
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(*signal.shape).astype(np.float32) * np.sqrt(noise_power)
    return signal + noise


def linear_resample(raw_signal: np.ndarray, target_len: int) -> np.ndarray:
    if len(raw_signal) == target_len:
        return raw_signal.astype(np.float32)
    resized = cv2.resize(raw_signal.reshape(1, -1), (target_len, 1), interpolation=cv2.INTER_LINEAR)
    return resized.flatten().astype(np.float32)


def decimate_resample(raw_signal: np.ndarray, target_len: int) -> np.ndarray:
    if len(raw_signal) <= target_len:
        return linear_resample(raw_signal, target_len)
    idx = np.linspace(0, len(raw_signal) - 1, target_len, dtype=np.int64)
    return raw_signal[idx].astype(np.float32)


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def make_spectrogram_rgb(signal: np.ndarray, image_size: tuple[int, int], fs: int, nperseg: int, overlap_ratio: float) -> np.ndarray:
    h, w = image_size
    noverlap = int(max(0, min(nperseg - 1, round(nperseg * overlap_ratio))))
    _, _, zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    spec = np.log(np.abs(zxx) + 1e-6)
    spec = cv2.resize(normalize_minmax(spec), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.stack([spec, spec, spec], axis=-1).astype(np.float32)


def preprocess_signal(raw_chunk: np.ndarray, cfg: dict) -> np.ndarray:
    data_cfg = cfg["data"]
    mode = str(data_cfg.get("signal_preprocess", "linear")).lower()
    target_len = int(data_cfg.get("target_signal_len", 8192))

    if mode == "linear":
        sig = linear_resample(raw_chunk, target_len)
    elif mode == "decimate":
        sig = decimate_resample(raw_chunk, target_len)
    elif mode == "identity_crop":
        if len(raw_chunk) >= target_len:
            sig = raw_chunk[:target_len].astype(np.float32)
        else:
            sig = linear_resample(raw_chunk, target_len)
    else:
        raise ValueError(f"Unsupported data.signal_preprocess: {mode}")

    sig = sig - float(np.mean(sig))
    global_scale = float(data_cfg.get("global_scale", 1.0))
    sig = sig / (global_scale + 1e-6)
    return sig.astype(np.float32)


def build_image(signal: np.ndarray, cfg: dict) -> np.ndarray:
    data_cfg = cfg["data"]
    mode = str(data_cfg.get("image_preprocess", "none")).lower()
    h, w = tuple(data_cfg.get("image_size", [224, 224]))
    if mode == "none":
        return np.zeros((h, w, 3), dtype=np.float32)
    if mode == "spectrogram_rgb":
        fs = int(data_cfg.get("spectrogram_fs", data_cfg.get("sr", 1000000)))
        nperseg = int(data_cfg.get("stft_nperseg", 256))
        overlap_ratio = float(data_cfg.get("stft_overlap_ratio", 0.5))
        return make_spectrogram_rgb(signal, (h, w), fs=fs, nperseg=nperseg, overlap_ratio=overlap_ratio)
    raise ValueError(f"Unsupported data.image_preprocess: {mode}")


def augment_signal(sig: np.ndarray, aug_cfg: dict) -> np.ndarray:
    if not aug_cfg.get("use_augment", False):
        return sig
    p = aug_cfg.get("params", {})
    out = sig.copy()
    if random.random() < float(p.get("amplitude_scale_p", 0.0)):
        lo, hi = p.get("amplitude_scale_range", [0.9, 1.1])
        out *= random.uniform(float(lo), float(hi))
    if random.random() < float(p.get("noise_p", 0.0)):
        snr_lo, snr_hi = p.get("snr_db_range", [10.0, 30.0])
        out = add_awgn_numpy(out, random.uniform(float(snr_lo), float(snr_hi)))
    return out.astype(np.float32)


@dataclass
class SegmentMeta:
    path: str
    offset: int
    raw_len: int
    label: int


class BaselineDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train", return_meta: bool = False):
        self.cfg = cfg
        self.split = split
        self.is_train = split == "train"
        self.return_meta = return_meta

        data_cfg = cfg["data"]
        self.split_mode = str(data_cfg.get("split_mode", "temporal")).lower()
        self.split_ratio = data_cfg.get("split_ratio", [0.7, 0.15, 0.15])
        self.step_ratio = float(data_cfg.get("step_ratio", 0.1))

        target_len = int(data_cfg.get("target_signal_len", 8192))
        raw_window_len = int(data_cfg.get("raw_window_len", 0))
        base_factor = int(data_cfg.get("base_resample_factor", 150))
        self.raw_len = raw_window_len if raw_window_len > 0 else target_len * base_factor

        self.file_split_seed = int(data_cfg.get("file_split_seed", cfg["train"].get("seed", 42)))
        self.indices = self._build_indices()
        if self.is_train:
            random.shuffle(self.indices)

    def _resolve_targets(self):
        data_cfg = self.cfg["data"]
        fusion_map = data_cfg.get("fusion_map", {})
        if fusion_map:
            return list(fusion_map.keys()), {k: int(v) for k, v in fusion_map.items()}
        case_ids = data_cfg["case_ids"]
        return case_ids, {name: i for i, name in enumerate(case_ids)}

    def _iter_files(self, folder_name: str):
        data_dir = self.cfg["data"]["data_dir"]
        folder = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder):
            return []
        files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        if not self.cfg["data"].get("use_whitelist", False):
            return files
        allowed = set(self.cfg["data"].get("whitelist", []))
        return [f for f in files if os.path.basename(f) in allowed]

    def _file_split(self, files: list[str]) -> set[str]:
        files = list(files)
        rnd = random.Random(self.file_split_seed)
        rnd.shuffle(files)
        n = len(files)
        n_train = int(n * float(self.split_ratio[0]))
        n_val = int(n * float(self.split_ratio[1]))
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
        train_files = set(files[:n_train])
        val_files = set(files[n_train:n_train + n_val])
        test_files = set(files[n_train + n_val:])
        if self.split == "train":
            return train_files
        if self.split == "val":
            return val_files
        return test_files

    def _window_range(self, total_len: int):
        if self.split_mode == "file":
            return 0, total_len
        t0 = int(total_len * float(self.split_ratio[0]))
        t1 = int(total_len * float(self.split_ratio[0] + self.split_ratio[1]))
        if self.split == "train":
            return 0, t0
        if self.split == "val":
            return t0, t1
        return t1, total_len

    def _build_indices(self) -> list[SegmentMeta]:
        indices: list[SegmentMeta] = []
        folders, label_map = self._resolve_targets()
        for folder_name in folders:
            files = self._iter_files(folder_name)
            if not files:
                continue
            active_files = self._file_split(files) if self.split_mode == "file" else set(files)
            label = label_map[folder_name]
            for file_path in files:
                if file_path not in active_files:
                    continue
                try:
                    arr = np.load(file_path, mmap_mode="r")
                    total = len(arr)
                except Exception:
                    continue
                start, end = self._window_range(total)
                step = max(1, int(self.raw_len * self.step_ratio))
                if end - start < self.raw_len:
                    continue
                ptr = start
                while ptr + self.raw_len <= end:
                    indices.append(SegmentMeta(path=file_path, offset=ptr, raw_len=self.raw_len, label=label))
                    ptr += step
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        meta = self.indices[idx]
        raw = np.load(meta.path, mmap_mode="r")
        raw_chunk = np.array(raw[meta.offset:meta.offset + meta.raw_len], dtype=np.float32)

        sig = preprocess_signal(raw_chunk, self.cfg)
        force_snr = os.environ.get("FORCE_SNR")
        if force_snr:
            try:
                sig = add_awgn_numpy(sig, float(force_snr))
            except ValueError:
                pass

        if self.is_train:
            sig = augment_signal(sig, self.cfg.get("augment", {}).get("signal", {}))

        img = build_image(sig, self.cfg)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        sig_t = torch.from_numpy(sig).float().unsqueeze(0)
        lbl_t = int(meta.label)

        if self.return_meta:
            tag = f"{os.path.basename(meta.path)}_{meta.offset}"
            return img_t, sig_t, lbl_t, tag
        return img_t, sig_t, lbl_t


def _collate(batch):
    if not batch:
        return None
    if len(batch[0]) == 4:
        imgs, sigs, labels, tags = zip(*batch)
        return torch.stack(imgs), torch.stack(sigs), torch.tensor(labels, dtype=torch.long), list(tags)
    imgs, sigs, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(sigs), torch.tensor(labels, dtype=torch.long)


def create_dataloaders(cfg: dict, return_meta: bool = False):
    num_workers = int(cfg["data"].get("num_workers", 0))
    batch_size = int(cfg["data"].get("batch_size", 64))
    prefetch_factor = int(cfg["data"].get("prefetch_factor", 4))

    out = {}
    for split in ["train", "val", "test"]:
        ds = BaselineDataset(cfg, split=split, return_meta=return_meta)
        if len(ds) == 0:
            out[split] = {}
            continue
        out[split] = {
            "main": DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(num_workers > 0),
                prefetch_factor=(prefetch_factor if num_workers > 0 else None),
                collate_fn=_collate,
            )
        }
    return out["train"], out["val"], out["test"]


def run_offline_preprocessing(_cfg: dict):
    return None
