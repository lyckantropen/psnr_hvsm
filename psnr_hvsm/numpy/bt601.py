"""Functions for converting RGB to normalized BT.601 YCbCr and YPbPr."""

from typing import Tuple
import numpy as np

from . import validate_precision


def bt601ycbcr(a: np.ndarray, precision: str = '8-bit') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an RGB image into normalized YCbCr.

    The `precision` option selects how the YCbCr components are quantised:

    - ``'8-bit'`` (default) rounds each component to 8-bit code values before
      normalisation. This matches the 8-bit YCbCr pipeline of the reference
      implementation and reproduces the published TID2013 reference values
      this library is cross-checked against.
    - ``'float'`` keeps the full floating-point result of the same matrix
      arithmetic, without the rounding step.
    """
    validate_precision(precision)
    r, g, b = np.moveaxis(a, 2, 0).astype(np.float64)
    y = 16 + 65.481 * r / 255 + 128.553 * g / 255 + 24.966 * b / 255
    cb = 128 - 37.797 * r / 255 - 74.203 * g / 255 + 112.0 * b / 255
    cr = 128 + 112.0 * r / 255 - 93.786 * g / 255 - 18.214 * b / 255
    if precision == '8-bit':
        y = np.round(y) / 255
        cb = np.round(cb) / 255
        cr = np.round(cr) / 255
    else:
        y = y / 255
        cb = cb / 255
        cr = cr / 255
    return y, cb, cr


def bt601ypbpr(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a normalized RGB image into YPbPr (analog YCbCr)."""
    r, g, b = np.moveaxis(a, 2, 0).astype(np.float64)
    kb = 0.114
    kr = 0.299
    y = kr * r + (1 - kr - kb) * g + kb * b
    pb = -(kr / (2 - 2 * kb)) * r - (1 - kr - kb) / (2 - 2 * kb) * g + 0.5 * b + 0.5
    pr = 0.5 * r - (1 - kr - kb) / (2 - 2 * kr) * g - (kb / (2 - 2 * kr)) * b + 0.5
    return y, pb, pr
