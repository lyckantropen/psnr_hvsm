"""Definition of PSNR."""

import numpy as np


def get_psnr(s: float, mv: float) -> float:
    """Compute PSNR for MSE."""
    return np.where(s != 0.0, 10.0 * np.log10(np.divide(mv**2, s)), np.full_like(s, 100.0)).clip(0, 100.0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Peak Signal-to-Noise Ratio of two normalized signals."""
    mse = np.power(a - b, 2.).mean()
    return get_psnr(mse, 1.0)
