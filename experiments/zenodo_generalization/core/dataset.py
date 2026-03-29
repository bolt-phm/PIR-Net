import copy
import csv
import glob
import os
import random
import re
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import spectrogram
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

warnings.filterwarnings("ignore")

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


_SIGNAL_CACHE: dict[tuple, tuple[np.ndarray, int]] = {}


def add_awgn_numpy(signal: np.ndarray, snr_db: float | None) -> np.ndarray:
    if snr_db is None:
        return signal
    power = float(np.mean(signal ** 2))
    if power <= 1e-12:
        return signal
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(*signal.shape).astype(np.float32) * np.sqrt(noise_power)
    return (signal + noise).astype(np.float32)


def _flatten_numeric(v) -> np.ndarray | None:
    try:
        arr = np.asarray(v, dtype=np.float32).squeeze()
    except Exception:
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32)


def _parse_sample_rate(obj: dict, fallback: int) -> int:
    for k, v in obj.items():
        lk = str(k).lower()
        if ("sample" in lk and "rate" in lk) or lk in {"fs", "fsamp", "sampling_rate"}:
            arr = _flatten_numeric(v)
            if arr is not None and arr.size > 0:
                val = int(round(float(arr.reshape(-1)[0])))
                if val > 0:
                    return val
    return int(fallback)


def _resolve_channel_key(container_keys: set[str], channel: str, patterns: list[str]) -> str | None:
    for p in patterns:
        k = p.replace("{ch}", channel)
        if k in container_keys:
            return k
    # Fallback: loose matching
    target = channel.lower().replace("_", "")
    for k in container_keys:
        lk = k.lower().replace("_", "").replace(" ", "")
        if lk.endswith(target):
            return k
    return None


def _linear_resample_multichannel(x: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    if src_fs == dst_fs:
        return x
    c, n = x.shape
    new_n = max(1, int(round(n * float(dst_fs) / float(src_fs))))
    old_pos = np.linspace(0.0, 1.0, n, dtype=np.float32)
    new_pos = np.linspace(0.0, 1.0, new_n, dtype=np.float32)
    out = np.zeros((c, new_n), dtype=np.float32)
    for i in range(c):
        out[i] = np.interp(new_pos, old_pos, x[i]).astype(np.float32)
    return out


def _mat_to_channels(path: str, channels: list[str], channel_key_patterns: list[str], expected_fs: int, allow_resample: bool, target_fs: int) -> tuple[np.ndarray, int]:
    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        keys = {k for k in mat.keys() if not k.startswith("__")}
        series = []
        for ch in channels:
            key = _resolve_channel_key(keys, ch, channel_key_patterns)
            if key is None:
                raise KeyError(f"Channel '{ch}' not found in {os.path.basename(path)}")
            arr = _flatten_numeric(mat[key])
            if arr is None:
                raise ValueError(f"Channel '{ch}' cannot be parsed as numeric array in {os.path.basename(path)}")
            series.append(arr)
        min_len = min(len(a) for a in series)
        stacked = np.stack([a[:min_len] for a in series], axis=0).astype(np.float32)
        src_fs = _parse_sample_rate(mat, expected_fs)
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError("MAT v7.3 detected but h5py is not installed. Please run `pip install h5py`.")
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            series = []
            for ch in channels:
                key = _resolve_channel_key(keys, ch, channel_key_patterns)
                if key is None:
                    raise KeyError(f"Channel '{ch}' not found in {os.path.basename(path)}")
                arr = np.array(f[key]).astype(np.float32).squeeze()
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                series.append(arr)
            min_len = min(len(a) for a in series)
            stacked = np.stack([a[:min_len] for a in series], axis=0).astype(np.float32)
            src_fs = expected_fs
            for k in keys:
                lk = str(k).lower()
                if ("sample" in lk and "rate" in lk) or lk in {"fs", "fsamp", "sampling_rate"}:
                    try:
                        sr_v = float(np.array(f[k]).reshape(-1)[0])
                        if sr_v > 0:
                            src_fs = int(round(sr_v))
                            break
                    except Exception:
                        pass

    if allow_resample and src_fs != target_fs:
        stacked = _linear_resample_multichannel(stacked, src_fs=src_fs, dst_fs=target_fs)
        return stacked, target_fs
    return stacked, src_fs


def _parse_test_id(path: str, min_id: int, max_id: int) -> int | None:
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = [int(x) for x in re.findall(r"\d+", stem)]
    if not nums:
        return None
    valid = [v for v in nums if min_id <= v <= max_id]
    if valid:
        return valid[-1]
    # Last resort, if name contains only one number (e.g., 0017) and range is small
    last = nums[-1]
    if 0 <= last <= 9999:
        tail = int(str(last)[-2:])
        if min_id <= tail <= max_id:
            return tail
    return None


def _default_label_by_test_id() -> dict[str, int]:
    out = {}
    for i in range(1, 15):
        out[str(i)] = 0
    for i in range(15, 18):
        out[str(i)] = 1
    return out


def _normalize_signal(sig: np.ndarray, mode: str, global_scale: float) -> np.ndarray:
    if mode == "none":
        return sig.astype(np.float32)
    if mode == "global_scale":
        return (sig / (global_scale + 1e-6)).astype(np.float32)
    # per_channel_zscore
    mu = np.mean(sig, axis=1, keepdims=True)
    std = np.std(sig, axis=1, keepdims=True)
    return ((sig - mu) / (std + 1e-6)).astype(np.float32)


def _augment_signal(sig: np.ndarray, aug_cfg: dict) -> np.ndarray:
    if not aug_cfg.get("use_augment", False):
        return sig
    p = aug_cfg.get("params", {})
    out = sig.copy()

    if random.random() < float(p.get("amplitude_scale_p", 0.0)):
        lo, hi = p.get("amplitude_scale_range", [0.9, 1.1])
        out *= random.uniform(float(lo), float(hi))

    if random.random() < float(p.get("noise_p", 0.0)):
        snr_lo, snr_hi = p.get("snr_db_range", [10.0, 30.0])
        noise_snr = random.uniform(float(snr_lo), float(snr_hi))
        out = add_awgn_numpy(out, noise_snr)

    return out.astype(np.float32)


def _resize_2d(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(out_hw[0]), int(out_hw[1])
    ten = torch.from_numpy(arr.astype(np.float32))[None, None, :, :]
    out = torch.nn.functional.interpolate(ten, size=(h, w), mode="bilinear", align_corners=False)
    return out[0, 0].cpu().numpy().astype(np.float32)


def _normalize_map01(arr: np.ndarray) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _compress_restore_1d(x: np.ndarray, mode: str, ratio: int, alpha: float) -> np.ndarray:
    n = int(len(x))
    if n <= 2 or ratio <= 1 or mode == "none":
        return x.astype(np.float32)

    target = max(2, n // int(ratio))
    chunks = np.array_split(x, target)
    comp = np.zeros((len(chunks),), dtype=np.float32)
    for i, ch in enumerate(chunks):
        if ch.size == 0:
            comp[i] = 0.0
            continue
        if mode == "average":
            v = float(np.mean(ch))
        elif mode == "max":
            idx = int(np.argmax(np.abs(ch)))
            v = float(ch[idx])
        elif mode == "decimate":
            v = float(ch[0])
        elif mode == "physics":
            # Physics-aware proxy: preserve sparse impulsive peaks while keeping the trend term.
            idx = int(np.argmax(np.abs(ch)))
            peak = float(ch[idx])
            mean = float(np.mean(ch))
            v = float((1.0 - alpha) * mean + alpha * peak)
        else:
            v = float(np.mean(ch))
        comp[i] = v

    src_pos = np.linspace(0.0, 1.0, len(comp), dtype=np.float32)
    dst_pos = np.linspace(0.0, 1.0, n, dtype=np.float32)
    restored = np.interp(dst_pos, src_pos, comp).astype(np.float32)
    return restored


@dataclass
class SegmentMeta:
    path: str
    test_id: int
    label: int
    start: int
    length: int


class ZenodoDynamicDataset(Dataset):
    def __init__(self, cfg: dict, split: str, return_meta: bool = False, override_test_ids: list[int] | None = None):
        self.cfg = cfg
        self.split = split
        self.return_meta = return_meta
        self.is_train = split == "train"

        data_cfg = cfg["data"]
        self.data_dir = os.path.abspath(data_cfg["data_dir"])
        self.channels = list(data_cfg.get("channels", ["A", "B", "C", "D", "E", "F", "Gx", "Gy"]))
        self.channel_key_patterns = list(
            data_cfg.get(
                "channel_key_patterns",
                ["Data1_{ch}", "Data1 {ch}", "Data1{ch}", "{ch}", "{ch}_data", "ch_{ch}"],
            )
        )
        self.sample_rate = int(data_cfg.get("sample_rate", 500))
        self.expected_sample_rate = int(data_cfg.get("expected_sample_rate", self.sample_rate))
        self.allow_resample = bool(data_cfg.get("allow_resample", True))
        self.window_seconds = float(data_cfg.get("window_seconds", 4.0))
        self.stride_seconds = float(data_cfg.get("stride_seconds", 1.0))
        self.window_len = max(4, int(round(self.window_seconds * self.sample_rate)))
        self.step_len = max(1, int(round(self.stride_seconds * self.sample_rate)))
        self.min_id = int(data_cfg.get("test_id_min", 1))
        self.max_id = int(data_cfg.get("test_id_max", 17))
        self.apply_sign = bool(data_cfg.get("apply_config_sign_correction", True))
        self.test_cfg_map = {int(k): str(v) for k, v in data_cfg.get("test_config_by_id", {}).items()}
        self.sign_map = {str(k): np.asarray(v, dtype=np.float32).reshape(-1) for k, v in data_cfg.get("sign_by_config", {}).items()}
        self.label_map = {str(k): int(v) for k, v in data_cfg.get("label_by_test_id", _default_label_by_test_id()).items()}
        self.norm_mode = str(data_cfg.get("normalization", "per_channel_zscore")).lower()
        self.global_scale = float(data_cfg.get("global_scale", 1.0))
        h, w = data_cfg.get("image_size", [32, 32])
        self.image_size = (int(h), int(w))
        self.image_out_channels = int(data_cfg.get("image_out_channels", 3))
        self.use_pseudo_image = bool(data_cfg.get("use_pseudo_image", False))
        # For signal-only models, optionally skip image tensor creation/transfers entirely.
        self.omit_image_tensor_when_disabled = bool(data_cfg.get("omit_image_tensor_when_disabled", False))
        self.image_mode = str(data_cfg.get("image_mode", "stft_rgb")).lower()
        self.pseudo_image_channels = [int(x) for x in data_cfg.get("pseudo_image_channels", [0, 1, 2])]
        self.stft_n_fft = int(data_cfg.get("stft_n_fft", 128))
        self.stft_hop_length = int(data_cfg.get("stft_hop_length", 32))

        res_cfg = data_cfg.get("resampling", {})
        self.resampling_enabled = bool(res_cfg.get("enabled", False))
        self.resampling_mode = str(res_cfg.get("mode", "none")).lower()
        self.resampling_ratio = int(res_cfg.get("compression_ratio", 1))
        self.resampling_alpha = float(res_cfg.get("alpha", 0.7))
        self.signal_aug_cfg = self.cfg.get("augment", {}).get("signal", {})

        # Optional RAM cache for high-throughput training/evaluation.
        # Default is disabled to preserve historical behavior.
        ram_cfg = data_cfg.get("ram_cache", {})
        self.ram_cache_enabled = bool(ram_cfg.get("enabled", False))
        self.ram_cache_dtype = str(ram_cfg.get("dtype", "float16")).lower()
        if self.ram_cache_dtype not in {"float16", "float32"}:
            self.ram_cache_dtype = "float16"
        self.ram_cache_log_interval = max(0, int(ram_cfg.get("log_interval", 500)))
        self.cache_train_signal = bool(ram_cfg.get("cache_train_signal", True))
        self.cache_train_image = bool(ram_cfg.get("cache_train_image", False))
        self.cache_eval_signal = bool(ram_cfg.get("cache_eval_signal", True))
        self.cache_eval_image = bool(ram_cfg.get("cache_eval_image", True))
        self.allow_train_image_cache_with_augmentation = bool(
            ram_cfg.get("allow_train_image_cache_with_augmentation", False)
        )
        self.train_uses_augmentation = bool(self.is_train and self.signal_aug_cfg.get("use_augment", False))

        self._cached_signal: np.ndarray | None = None
        self._cached_image: np.ndarray | None = None
        self._ram_cache_silent = str(os.environ.get("RAM_CACHE_SILENT", "0")).strip().lower() in {"1", "true", "yes", "on"}

        if override_test_ids is None:
            key = f"{split}_test_ids"
            if key not in data_cfg:
                raise KeyError(f"Missing data.{key} in config.")
            self.active_test_ids = {int(x) for x in data_cfg[key]}
        else:
            self.active_test_ids = {int(x) for x in override_test_ids}

        files = self._discover_files()
        self._signals: dict[str, np.ndarray] = {}
        self._labels: dict[str, int] = {}
        self._test_ids: dict[str, int] = {}
        self.indices: list[SegmentMeta] = []
        self.file_rows: list[dict] = []

        for fp in files:
            test_id = _parse_test_id(fp, self.min_id, self.max_id)
            if test_id is None or test_id not in self.active_test_ids:
                continue

            label = self.label_map.get(str(test_id), None)
            if label is None:
                continue

            sig = self._load_signal(fp, test_id)
            if sig.shape[1] < self.window_len:
                continue

            self._signals[fp] = sig
            self._labels[fp] = int(label)
            self._test_ids[fp] = int(test_id)

            n_seg = 0
            for start in range(0, sig.shape[1] - self.window_len + 1, self.step_len):
                self.indices.append(SegmentMeta(path=fp, test_id=int(test_id), label=int(label), start=int(start), length=self.window_len))
                n_seg += 1

            self.file_rows.append(
                {
                    "split": self.split,
                    "file_path": fp,
                    "file_name": os.path.basename(fp),
                    "test_id": int(test_id),
                    "label": int(label),
                    "n_channels": int(sig.shape[0]),
                    "n_samples": int(sig.shape[1]),
                    "n_segments": int(n_seg),
                }
            )

        # Keep deterministic segment order. DataLoader shuffle/sampler controls stochasticity.
        self._build_ram_cache_if_enabled()

    def _cache_dtype_np(self):
        return np.float16 if self.ram_cache_dtype == "float16" else np.float32

    def _build_ram_cache_if_enabled(self):
        if not self.ram_cache_enabled or len(self.indices) == 0:
            return

        split_cache_signal = (self.is_train and self.cache_train_signal) or ((not self.is_train) and self.cache_eval_signal)
        split_cache_image = (self.is_train and self.cache_train_image) or ((not self.is_train) and self.cache_eval_image)

        if self.is_train and split_cache_image and self.train_uses_augmentation and not self.allow_train_image_cache_with_augmentation:
            warnings.warn(
                "ram_cache.cache_train_image=True but train augmentation is enabled. "
                "To preserve strict image-signal consistency under augmentation, train image cache is disabled. "
                "Set ram_cache.allow_train_image_cache_with_augmentation=true to force-enable it.",
                RuntimeWarning,
            )
            split_cache_image = False

        split_cache_image = bool(split_cache_image and self.use_pseudo_image)
        if not split_cache_signal and not split_cache_image:
            return

        cache_dtype = self._cache_dtype_np()
        n = len(self.indices)
        c = len(self.channels)
        h, w = self.image_size
        ch = int(self.image_out_channels)

        if split_cache_signal:
            self._cached_signal = np.empty((n, c, self.window_len), dtype=cache_dtype)
        if split_cache_image:
            self._cached_image = np.empty((n, ch, h, w), dtype=cache_dtype)

        for i, m in enumerate(self.indices):
            seg = self._signals[m.path][:, m.start : m.start + m.length].copy().astype(np.float32)
            seg = self._apply_resampling(seg)

            if self._cached_signal is not None:
                self._cached_signal[i] = seg.astype(cache_dtype, copy=False)
            if self._cached_image is not None:
                img = self._build_pseudo_image(seg)
                self._cached_image[i] = img.astype(cache_dtype, copy=False)

            if (not self._ram_cache_silent) and self.ram_cache_log_interval > 0 and (i + 1) % self.ram_cache_log_interval == 0:
                print(f"[RAM CACHE] split={self.split} cached {i + 1}/{n} segments")

        sig_mb = 0.0 if self._cached_signal is None else float(self._cached_signal.nbytes / (1024 ** 2))
        img_mb = 0.0 if self._cached_image is None else float(self._cached_image.nbytes / (1024 ** 2))
        if not self._ram_cache_silent:
            print(
                f"[RAM CACHE] split={self.split} done | signal_cache={'on' if self._cached_signal is not None else 'off'} "
                f"| image_cache={'on' if self._cached_image is not None else 'off'} "
                f"| memory={sig_mb + img_mb:.1f} MiB (sig={sig_mb:.1f}, img={img_mb:.1f})"
            )

    def _discover_files(self) -> list[str]:
        patterns = self.cfg["data"].get(
            "file_glob",
            [
                "**/Dynamic*Random*/*Piezo*/*.mat",
                "**/*Dynamic*/*Piezo*/*.mat",
                "**/*Piezo*/*.mat",
            ],
        )
        out = []
        for p in patterns:
            out.extend(glob.glob(os.path.join(self.data_dir, p), recursive=True))
        # De-duplicate while preserving deterministic order
        return sorted(set(os.path.abspath(x) for x in out))

    def _load_signal(self, file_path: str, test_id: int) -> np.ndarray:
        cfg_key = ""
        if self.apply_sign:
            cfg_key = self.test_cfg_map.get(int(test_id), "")
        sign_vec = self.sign_map.get(cfg_key, None)
        sign_key = tuple(sign_vec.tolist()) if sign_vec is not None else ()
        cache_key = (
            os.path.abspath(file_path),
            tuple(self.channels),
            tuple(self.channel_key_patterns),
            int(self.sample_rate),
            int(self.expected_sample_rate),
            bool(self.allow_resample),
            cfg_key,
            sign_key,
        )
        if cache_key in _SIGNAL_CACHE:
            sig, _ = _SIGNAL_CACHE[cache_key]
            return sig

        sig, _sr = _mat_to_channels(
            path=file_path,
            channels=self.channels,
            channel_key_patterns=self.channel_key_patterns,
            expected_fs=self.expected_sample_rate,
            allow_resample=self.allow_resample,
            target_fs=self.sample_rate,
        )

        if self.apply_sign and sign_vec is not None and len(sign_vec) == sig.shape[0]:
            sig = sig * sign_vec.reshape(-1, 1)

        sig = _normalize_signal(sig, self.norm_mode, self.global_scale)
        _SIGNAL_CACHE[cache_key] = (sig.astype(np.float32), self.sample_rate)
        return sig.astype(np.float32)

    def _apply_resampling(self, seg: np.ndarray) -> np.ndarray:
        if (not self.resampling_enabled) or self.resampling_mode == "none" or self.resampling_ratio <= 1:
            return seg.astype(np.float32)
        out = np.zeros_like(seg, dtype=np.float32)
        for c in range(seg.shape[0]):
            out[c] = _compress_restore_1d(
                seg[c].astype(np.float32),
                mode=self.resampling_mode,
                ratio=self.resampling_ratio,
                alpha=self.resampling_alpha,
            )
        return out.astype(np.float32)

    def _channel_stft_map(self, x: np.ndarray) -> np.ndarray:
        n_fft = max(8, min(int(self.stft_n_fft), int(len(x))))
        hop = max(1, min(int(self.stft_hop_length), n_fft - 1))
        noverlap = max(0, n_fft - hop)
        _, _, sxx = spectrogram(
            x.astype(np.float32),
            fs=float(self.sample_rate),
            window="hann",
            nperseg=n_fft,
            noverlap=noverlap,
            nfft=n_fft,
            mode="magnitude",
            scaling="spectrum",
        )
        if sxx.size == 0:
            sxx = np.zeros((8, 8), dtype=np.float32)
        sxx = np.log1p(np.abs(sxx).astype(np.float32))
        sxx = _normalize_map01(sxx)
        return _resize_2d(sxx, self.image_size)

    def _build_pseudo_image(self, seg: np.ndarray) -> np.ndarray:
        if not self.use_pseudo_image:
            h, w = self.image_size
            return np.zeros((self.image_out_channels, h, w), dtype=np.float32)

        valid_idx = [i for i in self.pseudo_image_channels if 0 <= int(i) < seg.shape[0]]
        if not valid_idx:
            valid_idx = list(range(min(seg.shape[0], 3)))
        if not valid_idx:
            h, w = self.image_size
            return np.zeros((self.image_out_channels, h, w), dtype=np.float32)

        maps: list[np.ndarray] = []
        if self.image_mode == "five_channel":
            base = seg[valid_idx[0]]
            maps.append(self._channel_stft_map(base))
            maps.append(self._channel_stft_map(np.abs(np.diff(base, prepend=base[:1]))))
            maps.append(self._channel_stft_map(np.abs(base)))
            maps.append(self._channel_stft_map(seg[valid_idx[min(1, len(valid_idx) - 1)]]))
            maps.append(self._channel_stft_map(seg[valid_idx[min(2, len(valid_idx) - 1)]]))
        else:
            # Default: STFT pseudo-RGB maps from selected channels.
            for idx in valid_idx[: max(3, self.image_out_channels)]:
                maps.append(self._channel_stft_map(seg[int(idx)]))

        if not maps:
            h, w = self.image_size
            return np.zeros((self.image_out_channels, h, w), dtype=np.float32)

        while len(maps) < self.image_out_channels:
            maps.append(maps[len(maps) % len(maps)])
        maps = maps[: self.image_out_channels]
        return np.stack(maps, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        m = self.indices[idx]
        if self._cached_signal is not None:
            seg = self._cached_signal[idx].astype(np.float32, copy=True)
        else:
            sig_full = self._signals[m.path]
            seg = sig_full[:, m.start : m.start + m.length].copy().astype(np.float32)
            seg = self._apply_resampling(seg)

        force_snr = os.environ.get("FORCE_SNR")
        force_snr_value = None
        if force_snr:
            try:
                force_snr_value = float(force_snr)
                seg = add_awgn_numpy(seg, force_snr_value)
            except ValueError:
                pass

        if self.is_train:
            seg = _augment_signal(seg, self.signal_aug_cfg)

        if (not self.use_pseudo_image) and self.omit_image_tensor_when_disabled:
            img_t = None
        else:
            # Important: if FORCE_SNR is set (noisy eval), image must be recomputed from noisy signal.
            use_cached_image = (self._cached_image is not None) and (force_snr_value is None)
            if use_cached_image:
                img = self._cached_image[idx].astype(np.float32, copy=False)
            else:
                img = self._build_pseudo_image(seg)
            img_t = torch.from_numpy(img)

        sig_t = torch.from_numpy(seg)
        lbl_t = int(m.label)

        if self.return_meta:
            tag = f"{os.path.basename(m.path)}::t{m.test_id}::s{m.start}"
            return img_t, sig_t, lbl_t, tag
        return img_t, sig_t, lbl_t

    def summary(self) -> dict:
        by_label = {}
        by_test = {}
        for m in self.indices:
            by_label[m.label] = by_label.get(m.label, 0) + 1
            by_test[m.test_id] = by_test.get(m.test_id, 0) + 1
        return {
            "split": self.split,
            "n_files": len(self.file_rows),
            "n_segments": len(self.indices),
            "segments_by_label": by_label,
            "segments_by_test_id": dict(sorted(by_test.items())),
        }

    def export_manifest(self, csv_path: str):
        if not self.file_rows:
            return
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        keys = list(self.file_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.file_rows)


def _collate(batch):
    if not batch:
        return None

    def _stack_images_or_none(imgs):
        has_none = any(x is None for x in imgs)
        if not has_none:
            return torch.stack(imgs)
        if not all(x is None for x in imgs):
            raise RuntimeError("Mixed None/non-None image tensors in one batch.")
        return None

    if len(batch[0]) == 4:
        imgs, sigs, labels, tags = zip(*batch)
        return _stack_images_or_none(imgs), torch.stack(sigs), torch.tensor(labels, dtype=torch.long), list(tags)
    imgs, sigs, labels = zip(*batch)
    return _stack_images_or_none(imgs), torch.stack(sigs), torch.tensor(labels, dtype=torch.long)


def _iter_segment_meta(ds):
    if isinstance(ds, Subset):
        base = ds.dataset
        if hasattr(base, "indices"):
            for i in ds.indices:
                yield base.indices[int(i)]
        return
    if hasattr(ds, "indices"):
        for m in ds.indices:
            yield m


def _labels_from_dataset(ds) -> list[int]:
    labels = []
    for m in _iter_segment_meta(ds):
        labels.append(int(m.label))
    return labels


def _stratified_split_indices(labels: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if len(labels) == 0:
        return [], []
    y = np.asarray(labels, dtype=np.int64)
    all_idx = np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    train_idx = []
    val_idx = []
    for cls in sorted(set(y.tolist())):
        cls_idx = all_idx[y == cls].copy()
        rng.shuffle(cls_idx)
        if len(cls_idx) == 1:
            val_n = 0
        else:
            val_n = int(round(len(cls_idx) * float(val_ratio)))
            val_n = max(1, val_n)
            val_n = min(val_n, len(cls_idx) - 1)
        val_idx.extend(cls_idx[:val_n].tolist())
        train_idx.extend(cls_idx[val_n:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return [int(x) for x in train_idx], [int(x) for x in val_idx]


def dataset_summary(ds, split_name: str = "unknown") -> dict:
    by_label = {}
    by_test = {}
    file_set = set()
    n_segments = 0
    for m in _iter_segment_meta(ds):
        n_segments += 1
        by_label[int(m.label)] = by_label.get(int(m.label), 0) + 1
        by_test[int(m.test_id)] = by_test.get(int(m.test_id), 0) + 1
        file_set.add(str(m.path))
    return {
        "split": split_name,
        "n_files": len(file_set),
        "n_segments": n_segments,
        "segments_by_label": by_label,
        "segments_by_test_id": dict(sorted(by_test.items())),
    }


def dataset_export_manifest(ds, csv_path: str, split_name: str = "unknown"):
    rows = {}
    base = ds.dataset if isinstance(ds, Subset) else ds
    for m in _iter_segment_meta(ds):
        key = str(m.path)
        if key not in rows:
            n_channels = int(base._signals[key].shape[0]) if hasattr(base, "_signals") and key in base._signals else -1
            n_samples = int(base._signals[key].shape[1]) if hasattr(base, "_signals") and key in base._signals else -1
            rows[key] = {
                "split": split_name,
                "file_path": key,
                "file_name": os.path.basename(key),
                "test_id": int(m.test_id),
                "label": int(m.label),
                "n_channels": n_channels,
                "n_samples": n_samples,
                "n_segments": 0,
            }
        rows[key]["n_segments"] += 1

    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    keys = list(next(iter(rows.values())).keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in sorted(rows.values(), key=lambda x: x["file_name"]):
            w.writerow(row)


def _make_loader(cfg: dict, split_name: str, ds):
    data_cfg = cfg["data"]
    default_workers = int(data_cfg.get("num_workers", 4))
    default_batch = int(data_cfg.get("batch_size", 256))
    if split_name == "train":
        num_workers = int(data_cfg.get("num_workers_train", default_workers))
        batch_size = int(data_cfg.get("batch_size_train", default_batch))
    else:
        num_workers = int(data_cfg.get("num_workers_eval", default_workers))
        batch_size = int(data_cfg.get("batch_size_eval", default_batch))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4))

    if len(ds) == 0:
        return {}

    sampler = None
    shuffle = split_name == "train"
    if split_name == "train" and bool(cfg.get("train", {}).get("balance_sampler", False)):
        labels = _labels_from_dataset(ds)
        class_counts = {}
        for y in labels:
            class_counts[y] = class_counts.get(y, 0) + 1
        weights = [1.0 / max(1, class_counts[y]) for y in labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False

    return {
        "main": DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            collate_fn=_collate,
        )
    }


def create_dataloaders(cfg: dict, return_meta: bool = False, return_datasets: bool = False, split_override: dict[str, list[int]] | None = None):
    data_cfg = cfg["data"]
    train_override = split_override.get("train") if split_override and "train" in split_override else None
    val_override = split_override.get("val") if split_override and "val" in split_override else None
    test_override = split_override.get("test") if split_override and "test" in split_override else None

    train_ids = [int(x) for x in (train_override if train_override is not None else data_cfg.get("train_test_ids", []))]
    val_ids = [int(x) for x in (val_override if val_override is not None else data_cfg.get("val_test_ids", []))]
    test_ids = [int(x) for x in (test_override if test_override is not None else data_cfg.get("test_test_ids", []))]

    train_ds = ZenodoDynamicDataset(cfg, split="train", return_meta=return_meta, override_test_ids=train_ids)
    test_ds = ZenodoDynamicDataset(cfg, split="test", return_meta=return_meta, override_test_ids=test_ids)

    use_val_from_train = bool(data_cfg.get("use_val_from_train", False)) or len(val_ids) == 0
    if use_val_from_train:
        val_ratio = float(data_cfg.get("val_from_train_ratio", 0.15))
        split_seed = int(data_cfg.get("val_from_train_seed", cfg.get("train", {}).get("seed", 42)))
        train_labels = _labels_from_dataset(train_ds)
        train_idx, val_idx = _stratified_split_indices(train_labels, val_ratio=val_ratio, seed=split_seed)

        # Build a non-augmented view with identical segment ordering for validation.
        train_as_val = ZenodoDynamicDataset(cfg, split="val", return_meta=return_meta, override_test_ids=train_ids)
        if len(train_as_val.indices) != len(train_ds.indices):
            raise RuntimeError("Internal split mismatch between train and val views. Please check deterministic dataset construction.")
        train_ds = Subset(train_ds, train_idx)
        val_ds = Subset(train_as_val, val_idx)
    else:
        val_ds = ZenodoDynamicDataset(cfg, split="val", return_meta=return_meta, override_test_ids=val_ids)

    train_loaders = _make_loader(cfg, "train", train_ds)
    val_loaders = _make_loader(cfg, "val", val_ds)
    test_loaders = _make_loader(cfg, "test", test_ds)

    if return_datasets:
        return train_loaders, val_loaders, test_loaders, train_ds, val_ds, test_ds
    return train_loaders, val_loaders, test_loaders


def create_test_loaders_only(cfg: dict, return_meta: bool = False, test_override: list[int] | None = None):
    data_cfg = cfg["data"]
    test_ids = [int(x) for x in (test_override if test_override is not None else data_cfg.get("test_test_ids", []))]
    test_ds = ZenodoDynamicDataset(cfg, split="test", return_meta=return_meta, override_test_ids=test_ids)
    return _make_loader(cfg, "test", test_ds)


def run_offline_preprocessing(_cfg: dict):
    return None


def make_eval_cfg(base_cfg: dict, test_ids: list[int]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["test_test_ids"] = [int(x) for x in test_ids]
    return cfg
