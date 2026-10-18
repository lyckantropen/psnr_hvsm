"""Functions for converting RGB to normalized BT.601 YCbCr and YPbPr."""

from typing import Tuple

import numpy as np
import torch

PRECISIONS = ('8-bit', 'float')


def bt601ycbcr(a: torch.Tensor, precision: str = '8-bit') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert an RGB image into normalized YCbCr.

    The `precision` option selects how the YCbCr components are quantised:

    - ``'8-bit'`` (default) rounds each component to 8-bit code values before
      normalisation. This matches the 8-bit YCbCr pipeline of the reference
      implementation and reproduces the published TID2013 reference values
      this library is cross-checked against.
    - ``'float'`` keeps the full floating-point result of the same matrix
      arithmetic, without the rounding step.

    The semantics are identical to the NumPy implementation of this function.
    """
    if precision not in PRECISIONS:
        raise ValueError("precision must be one of %r, got %r" % (PRECISIONS, precision))
    if isinstance(a, np.ndarray):
        a = torch.tensor(a).moveaxis(-1, -3)
    r = a[..., 0, :, :]
    g = a[..., 1, :, :]
    b = a[..., 2, :, :]
    y = 16 + 65.481 * r / 255 + 128.553 * g / 255 + 24.966 * b / 255
    cb = 128 - 37.797 * r / 255 - 74.203 * g / 255 + 112.0 * b / 255
    cr = 128 + 112.0 * r / 255 - 93.786 * g / 255 - 18.214 * b / 255
    if precision == '8-bit':
        y = torch.round(y) / 255
        cb = torch.round(cb) / 255
        cr = torch.round(cr) / 255
    else:
        y = y / 255
        cb = cb / 255
        cr = cr / 255
    return y, cb, cr


def bt601ypbpr(a: torch.Tensor, precision: str = '8-bit') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a normalized RGB image into YPbPr (analog YCbCr).

    This function assumes (N,H,W,C) channel ordering.

    The `precision` option is accepted for symmetry with `bt601ycbcr`; YPbPr
    is never rounded, so it has no effect on the result.
    """
    if precision not in PRECISIONS:
        raise ValueError("precision must be one of %r, got %r" % (PRECISIONS, precision))
    if isinstance(a, np.ndarray):
        a = torch.tensor(a).moveaxis(-1, -3)
    r = a[..., 0, :, :]
    g = a[..., 1, :, :]
    b = a[..., 2, :, :]
    kb = 0.114
    kr = 0.299
    y = kr * r + (1 - kr - kb) * g + kb * b
    pb = -(kr / (2 - 2 * kb)) * r - (1 - kr - kb) / (2 - 2 * kb) * g + 0.5 * b + 0.5
    pr = 0.5 * r - (1 - kr - kb) / (2 - 2 * kr) * g - (kb / (2 - 2 * kr)) * b + 0.5
    return y, pb, pr
