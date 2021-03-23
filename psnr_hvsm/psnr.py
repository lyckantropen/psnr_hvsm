import numpy as np


def psnr(a, b):
    """Compute the Peak Signal-to-Noise Ratio of two normalized signals."""

    mse = np.power(a - b, 2.).mean()

    if mse == 0:
        return 100.0
    else:
        return min(10.0 * np.log10(1.0/mse), 100.0)
