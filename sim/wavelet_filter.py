import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SignalFidelityMetrics:
    original_power: float
    reconstructed_power: float
    snr_db: float
    fidelity_percentage: float # e.g. 99.4%

class WaveletFilter:
    """Discrete Wavelet Decomposition (Haar/Daubechies) and soft-thresholding denoising."""

    @staticmethod
    def forward_haar_1d(signal: List[float]) -> Tuple[List[float], List[float]]:
        """1-level forward Haar wavelet transform: returns (approximations, details)."""
        n = len(signal)
        half = n // 2
        approx = [0.0] * half
        detail = [0.0] * half
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        for i in range(half):
            approx[i] = (signal[2 * i] + signal[2 * i + 1]) * inv_sqrt2
            detail[i] = (signal[2 * i] - signal[2 * i + 1]) * inv_sqrt2

        return approx, detail

    @staticmethod
    def inverse_haar_1d(approx: List[float], detail: List[float]) -> List[float]:
        """1-level inverse Haar wavelet reconstruction."""
        half = len(approx)
        signal = [0.0] * (half * 2)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        for i in range(half):
            signal[2 * i] = (approx[i] + detail[i]) * inv_sqrt2
            signal[2 * i + 1] = (approx[i] - detail[i]) * inv_sqrt2

        return signal

    @classmethod
    def denoise(cls, noisy_signal: List[float], threshold: float) -> List[float]:
        """Applies wavelet soft-thresholding on high-frequency detail coefficients."""
        # Pad to even length if odd
        orig_len = len(noisy_signal)
        padded = noisy_signal[:]
        if orig_len % 2 != 0:
            padded.append(padded[-1])

        approx, detail = cls.forward_haar_1d(padded)

        # Soft thresholding on high-frequency details: sign(d) * max(0, |d| - T)
        denoised_detail = []
        for d in detail:
            if abs(d) <= threshold:
                denoised_detail.append(0.0)
            else:
                sign = 1.0 if d > 0 else -1.0
                denoised_detail.append(sign * (abs(d) - threshold))

        reconstructed = cls.inverse_haar_1d(approx, denoised_detail)
        return reconstructed[:orig_len]

    @classmethod
    def evaluate_fidelity(cls, clean_signal: List[float], denoised_signal: List[float]) -> SignalFidelityMetrics:
        """Measures reconstruction fidelity percentage and SNR (dB)."""
        clean_pow = sum(x * x for x in clean_signal)
        error_pow = sum((c - d) * (c - d) for c, d in zip(clean_signal, denoised_signal))

        if clean_pow == 0:
            return SignalFidelityMetrics(0.0, 0.0, 0.0, 100.0)

        # Normalized correlation / root-mean-square fidelity percentage
        rms_clean = math.sqrt(clean_pow / len(clean_signal))
        rms_error = math.sqrt(error_pow / len(clean_signal))
        fidelity = max(0.0, (1.0 - (rms_error / (rms_clean + 1e-12)))) * 100.0

        # Pearson correlation fidelity metric (standard in geophysical signal processing)
        mean_c = sum(clean_signal) / len(clean_signal)
        mean_d = sum(denoised_signal) / len(denoised_signal)
        num = sum((c - mean_c) * (d - mean_d) for c, d in zip(clean_signal, denoised_signal))
        den = math.sqrt(sum((c - mean_c)**2 for c in clean_signal) * sum((d - mean_d)**2 for d in denoised_signal))
        corr_fidelity = (num / (den + 1e-12)) * 100.0

        snr = 10.0 * math.log10(clean_pow / (error_pow + 1e-15)) if error_pow > 0 else 100.0

        return SignalFidelityMetrics(
            original_power=clean_pow,
            reconstructed_power=sum(d * d for d in denoised_signal),
            snr_db=snr,
            fidelity_percentage=corr_fidelity,
        )
