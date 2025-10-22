import unittest

import numpy as np
from scipy import signal

try:
    import cuda_feat_gen
except Exception:  # pragma: no cover - module import is validated in tests
    cuda_feat_gen = None


@unittest.skipIf(cuda_feat_gen is None, "cuda_feat_gen module is not available")
class GPUFeatureGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not cuda_feat_gen.is_torch_available():
            raise unittest.SkipTest("PyTorch is not installed")
        if not cuda_feat_gen.is_available():
            raise unittest.SkipTest("CUDA device is not available for testing")

        fs = 40_000.0
        nperseg = 256
        noverlap = 128
        cls.generator = cuda_feat_gen.GPUFeatureGenerator(
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            psd_window="hann",
            spec_window=np.hanning(nperseg),
        )
        cls.fs = fs
        cls.nperseg = nperseg
        cls.noverlap = noverlap

    def _sample_signals(self, length: int = 4096) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal(length).astype(np.float32)

    def test_welch_matches_cpu(self) -> None:
        samples = self._sample_signals()
        freqs_cpu, psd_cpu = signal.welch(
            samples,
            self.fs,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
        )

        freqs_gpu, psd_gpu_tensor = self.generator.welch(samples)
        self.generator.synchronize()
        psd_gpu = self.generator.to_numpy(psd_gpu_tensor)[0]
        freqs_gpu_np = self.generator.to_numpy(freqs_gpu)

        np.testing.assert_allclose(freqs_gpu_np, freqs_cpu, atol=1e-5)
        np.testing.assert_allclose(psd_gpu, psd_cpu, rtol=5e-2, atol=1e-6)

    def test_spectrogram_matches_cpu(self) -> None:
        samples = self._sample_signals()
        f_cpu, t_cpu, sxx_cpu = signal.spectrogram(
            samples,
            self.fs,
            window=np.hanning(self.nperseg),
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling='density',
            mode='psd',
        )

        f_gpu, t_gpu, sxx_gpu_tensor = self.generator.spectrogram(samples)
        self.generator.synchronize()
        f_gpu_np = self.generator.to_numpy(f_gpu)
        t_gpu_np = self.generator.to_numpy(t_gpu)
        sxx_gpu = self.generator.to_numpy(sxx_gpu_tensor)[0]

        np.testing.assert_allclose(f_gpu_np, f_cpu, atol=1e-5)
        np.testing.assert_allclose(t_gpu_np, t_cpu, atol=1e-5, rtol=1e-3)
        np.testing.assert_allclose(sxx_gpu, sxx_cpu, rtol=5e-2, atol=1e-6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
