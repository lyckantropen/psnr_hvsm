import numpy as np
from scipy.fft import dctn
from typing import Tuple

MASK_COEFF = np.array([0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874,
                       0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058,
                       0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888,
                       0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015,
                       0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866,
                       0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815,
                       0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803,
                       0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203], dtype=np.float64)

CSF_COEFF = np.array([1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887,
                      2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911,
                      1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555,
                      1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082,
                      1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222,
                      1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729,
                      0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803,
                      0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950], dtype=np.float64)

DCT_H, DCT_W = 8, 8


def get_psnr(s: np.ndarray, mv: float) -> np.ndarray:
    return np.where(s != 0.0, 10.0 * np.log10(np.divide(mv**2, s)), np.full_like(s, 100.0))


def to_blocks(x: np.ndarray) -> np.ndarray:
    shape = x.shape
    prev_shape = shape[:-2]
    h, w = shape[-2:]
    return np.moveaxis(x.reshape(-1, h // DCT_H, DCT_H, w // DCT_W, DCT_W), -2, -3).reshape(*prev_shape, -1, DCT_H, DCT_W)


def masking(tiles: np.ndarray, tiles_dct: np.ndarray) -> np.ndarray:
    qh = DCT_H // 2
    qw = DCT_W // 2

    acs = tiles_dct.reshape(*tiles_dct.shape[:-2], DCT_H * DCT_W)[..., 1:]
    mask = np.sum(np.power(acs, 2.0) * MASK_COEFF[1:], axis=-1)

    def vari(a: np.ndarray) -> np.ndarray:
        return np.var(a, axis=(-1, -2), ddof=1) * a.shape[-1] * a.shape[-2]

    var = vari(tiles)
    var = np.where(var != 0.0, (vari(tiles[..., :qh, :qw]) +
                                vari(tiles[..., :qh, qw:]) +
                                vari(tiles[..., qh:, :qw]) +
                                vari(tiles[..., qh:, qw:])) / var, 0.0)

    return np.sqrt(mask * var / (qh * qw) / (DCT_H * DCT_W))


def hvs_hvsm_mse_tiles(images_a: np.ndarray, images_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tiles_a = to_blocks(images_a)
    tiles_b = to_blocks(images_b)

    mask_coeff = MASK_COEFF.reshape((DCT_H, DCT_W))
    coeff = CSF_COEFF.reshape((DCT_H, DCT_W))
    dct_a = dctn(tiles_a, norm='ortho', axes=(-1, -2))
    dct_b = dctn(tiles_b, norm='ortho', axes=(-1, -2))

    dif = np.abs(dct_a - dct_b)
    mask_a = masking(tiles_a, dct_a)
    mask_b = masking(tiles_b, dct_b)

    mask_a = np.where(mask_b > mask_a, mask_b, mask_a)

    common_shape = (*mask_a.shape, *mask_coeff.shape)
    mask_coeff = np.broadcast_to(mask_coeff, common_shape)
    mask_a = np.broadcast_to(mask_a.reshape(*mask_a.shape, 1, 1), common_shape)

    weighted_mse = np.power(dif[..., 0, 0] * coeff[0, 0], 2.0)
    mask = np.power(mask_coeff, -1.0) * mask_a
    masked_dct = np.where(dif >= mask, dif - mask, 0.0) * coeff
    masked_dct_sq = np.power(masked_dct.reshape(*masked_dct.shape[:-2], DCT_H * DCT_W)[..., 1:], 2.0)
    weighted_mse += np.sum(masked_dct_sq, axis=-1)

    hvs_tiles = np.power(dif * coeff, 2.0) / (DCT_H * DCT_W)
    return (np.sum(hvs_tiles.reshape(*dif.shape[:-2], DCT_H * DCT_W), axis=-1),
            weighted_mse / (DCT_H * DCT_W))


def hvs_hvsm_mse(images_a: np.ndarray, images_b: np.ndarray, batch=False) -> Tuple[np.ndarray, np.ndarray]:
    hvs_tiles, hvsm_tiles = hvs_hvsm_mse_tiles(images_a, images_b)

    if batch or len(hvs_tiles.shape) < 2:
        return hvs_tiles.mean(axis=-1), hvsm_tiles.mean(axis=-1)
    else:
        return hvs_tiles.mean(axis=(0, -1)), hvsm_tiles.mean(axis=(0, -1))


def psnr_hvs_hvsm(images_a: np.ndarray, images_b: np.ndarray, batch=False) -> Tuple[np.ndarray, np.ndarray]:
    hvs_tiles, hvsm_tiles = hvs_hvsm_mse_tiles(images_a, images_b)

    if batch or len(hvs_tiles.shape) < 2:
        return get_psnr(hvs_tiles.mean(axis=-1), 1.0), get_psnr(hvsm_tiles.mean(axis=-1), 1.0)
    else:
        return get_psnr(hvs_tiles.mean(axis=-1), 1.0).mean(axis=0), get_psnr(hvsm_tiles.mean(axis=-1), 1.0).mean(axis=0)
