"""GPU-accelerated feature generation helpers for DroneRF pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional at runtime
    torch = None  # type: ignore


def is_torch_available() -> bool:
    """Return ``True`` when PyTorch can be imported."""

    return torch is not None


def is_available() -> bool:
    """Return ``True`` when CUDA-backed PyTorch tensors can be created."""

    return torch is not None and torch.cuda.is_available()


def _ensure_torch(message: str = "PyTorch is required for GPU feature generation") -> None:
    if torch is None:
        raise RuntimeError(message)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _prepare_window(
    window: Sequence[float],
    nperseg: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    if isinstance(window, torch.Tensor):
        tensor = window.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(window, dtype=dtype, device=device)
    if tensor.numel() != nperseg:
        raise ValueError(
            f"window length mismatch: expected {nperseg}, received {tensor.numel()}"
        )
    return tensor


def _create_window_from_name(
    name: str,
    nperseg: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    name = name.lower()
    if name in {"hann", "hanning"}:
        return torch.hann_window(nperseg, periodic=True, dtype=dtype, device=device)
    if name == "hamming":
        return torch.hamming_window(nperseg, periodic=True, dtype=dtype, device=device)
    if name == "blackman":
        return torch.blackman_window(nperseg, periodic=True, dtype=dtype, device=device)
    raise ValueError(f"Unsupported window type '{name}' for GPU feature generation")


@dataclass
class GPUFeatureGenerator:
    """GPU-backed helper that mirrors the SciPy feature generation APIs."""

    fs: float
    nperseg: int
    noverlap: int
    psd_window: Sequence[float] | str
    spec_window: Sequence[float]
    device: Optional[str] = None
    dtype: Optional["torch.dtype"] = None

    def __post_init__(self) -> None:
        _ensure_torch()
        _validate_positive("nperseg", self.nperseg)
        if self.noverlap < 0 or self.noverlap >= self.nperseg:
            raise ValueError("noverlap must be in the range [0, nperseg)")

        requested_device = torch.device(self.device) if self.device else torch.device("cuda")
        if requested_device.type != "cuda":
            raise ValueError(
                f"GPUFeatureGenerator requires a CUDA device; received '{requested_device}'"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for GPU feature generation")

        self.device = str(requested_device)
        self._device = requested_device
        self.dtype = self.dtype or torch.float32
        self.step = self.nperseg - self.noverlap

        if isinstance(self.psd_window, str):
            psd_tensor = _create_window_from_name(self.psd_window, self.nperseg, self._device, self.dtype)
        else:
            psd_tensor = _prepare_window(self.psd_window, self.nperseg, self._device, self.dtype)
        self._psd_window = psd_tensor
        self._psd_window_power = float(torch.sum(psd_tensor ** 2).item())

        self._spec_window = _prepare_window(self.spec_window, self.nperseg, self._device, self.dtype)
        self._spec_window_power = float(torch.sum(self._spec_window ** 2).item())

        self._freqs = torch.fft.rfftfreq(self.nperseg, d=1.0 / self.fs).to(
            device=self._device, dtype=self.dtype
        )

    def _ensure_batch_tensor(self, samples: np.ndarray | "torch.Tensor") -> "torch.Tensor":
        if isinstance(samples, torch.Tensor):
            tensor = samples.to(device=self._device, dtype=self.dtype, non_blocking=True)
        else:
            tensor = torch.as_tensor(samples, dtype=self.dtype, device=self._device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError("Input samples must be 1D or 2D (batch, samples)")
        return tensor

    def _segmentize(self, tensor: "torch.Tensor") -> "torch.Tensor":
        segments = tensor.unfold(-1, self.nperseg, self.step)
        if segments.size(-2) <= 0:
            raise ValueError(
                "Insufficient samples to form even a single frame for the given configuration"
            )
        return segments

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def welch(self, samples: np.ndarray | "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        batch_tensor = self._ensure_batch_tensor(samples)
        frames = self._segmentize(batch_tensor) * self._psd_window
        fft_values = torch.fft.rfft(frames, dim=-1)
        psd = (fft_values.abs() ** 2) / (self.fs * self._psd_window_power)
        psd = psd.mean(dim=-2)
        return self._freqs, psd

    def spectrogram(
        self,
        samples: np.ndarray | "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        batch_tensor = self._ensure_batch_tensor(samples)
        frames = self._segmentize(batch_tensor) * self._spec_window
        fft_values = torch.fft.rfft(frames, dim=-1)
        sxx = (fft_values.abs() ** 2) / (self.fs * self._spec_window_power)
        sxx = sxx.transpose(-2, -1)
        times = (
            torch.arange(sxx.size(-1), device=self._device, dtype=self.dtype) * self.step
            + (self.nperseg / 2)
        ) / self.fs
        return self._freqs, times, sxx

    def to_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
        return tensor.detach().to(device="cpu").numpy()


def welch(
    samples: np.ndarray | "torch.Tensor",
    fs: float,
    window: Sequence[float] | str,
    nperseg: int,
    noverlap: int,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    generator = GPUFeatureGenerator(fs, nperseg, noverlap, window, window, device=device)
    freqs, psd = generator.welch(samples)
    generator.synchronize()
    return generator.to_numpy(freqs), generator.to_numpy(psd)


def spectrogram(
    samples: np.ndarray | "torch.Tensor",
    fs: float,
    window: Sequence[float],
    nperseg: int,
    noverlap: int,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = GPUFeatureGenerator(fs, nperseg, noverlap, window, window, device=device)
    freqs, times, sxx = generator.spectrogram(samples)
    generator.synchronize()
    return generator.to_numpy(freqs), generator.to_numpy(times), generator.to_numpy(sxx)
