"""Functions for computing luma from a BGR signal."""

import numpy as np


def bt601luma(a):
    """Calculate the luma component of a BGR signal according to the BT.601 specification."""
    r, g, b = np.moveaxis(np.flip(a, 2), 2, 0).astype(np.float64)
    return np.round(16 + 65.481*r/255 + 128.553*g/255 + 24.966*b/255)/255


def bt601luma_norm(a):
    """Calculate the luma component of a BGR signal according to the BT.601 specification (assuming analog input)."""
    r, g, b = np.moveaxis(np.flip(a, 2), 2, 0).astype(np.float64)
    return 0.299*r + 0.587*g + 0.114*b
