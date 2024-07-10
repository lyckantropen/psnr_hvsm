from typing import Tuple
import torch
import torch_dct as dct
import numpy as np

from ..numpy.psnr_hvsm import MASK_COEFF, CSF_COEFF, DCT_H, DCT_W
from .psnr import get_psnr


def to_blocks(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    prev_shape = shape[:-2]
    h, w = shape[-2:]
    return x.view(-1, h // DCT_H, DCT_H, w // DCT_W, DCT_W).moveaxis(-2, -3).reshape(*prev_shape, -1, DCT_H, DCT_W)


def masking(tiles: torch.Tensor, tiles_dct: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    device = tiles.device
    qh = DCT_H // 2
    qw = DCT_W // 2
    mask_coeff = torch.tensor(MASK_COEFF, device=device)

    acs = tiles_dct.reshape(*tiles_dct.shape[:-2], DCT_H * DCT_W)[..., 1:]
    mask = torch.sum(torch.pow(acs, 2.0) * mask_coeff[1:], dim=-1)

    def vari(a: torch.Tensor) -> torch.Tensor:
        return torch.var(a, dim=(-1, -2)) * a.size(-1) * a.size(-2)

    var = vari(tiles)
    var = torch.where(var != 0.0, (vari(tiles[..., :qh, :qw]) +
                                   vari(tiles[..., :qh, qw:]) +
                                   vari(tiles[..., qh:, :qw]) +
                                   vari(tiles[..., qh:, qw:])) / (var + epsilon), 0.0)

    return torch.sqrt(mask * var / (qh * qw) / (DCT_H * DCT_W) + epsilon)


def hvs_hvsm_mse_tiles(images_a: torch.Tensor, images_b: torch.Tensor, masking_epsilon: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(images_a, np.ndarray):
        images_a = torch.tensor(images_a)
    if isinstance(images_b, np.ndarray):
        images_b = torch.tensor(images_b)

    tiles_a = to_blocks(images_a)
    tiles_b = to_blocks(images_b)

    device = tiles_a.device
    mask_coeff = torch.tensor(MASK_COEFF, device=device).reshape((DCT_H, DCT_W))
    coeff = torch.tensor(CSF_COEFF, device=device).reshape((DCT_H, DCT_W))
    dct_a = dct.dct_2d(tiles_a, norm='ortho')
    dct_b = dct.dct_2d(tiles_b, norm='ortho')

    dif = torch.abs(dct_a - dct_b)
    mask_a = masking(tiles_a, dct_a, epsilon=masking_epsilon)
    mask_b = masking(tiles_b, dct_b, epsilon=masking_epsilon)

    mask_a = torch.where(mask_b > mask_a, mask_b, mask_a)

    common_shape = (*mask_a.shape, *mask_coeff.shape)
    mask_coeff = torch.broadcast_to(mask_coeff, common_shape)
    mask_a = torch.broadcast_to(mask_a.reshape(*mask_a.shape, 1, 1), common_shape)

    weighted_mse = torch.pow(dif[..., 0, 0] * coeff[0, 0], 2.0)
    mask = torch.pow(mask_coeff, -1.0) * mask_a
    masked_dct = torch.where(dif >= mask, dif - mask, 0.0) * coeff
    masked_dct_sq = torch.pow(masked_dct.reshape(*masked_dct.shape[:-2], DCT_H * DCT_W)[..., 1:], 2.0)
    weighted_mse += torch.sum(masked_dct_sq, dim=-1)

    hvs_tiles = torch.pow(dif * coeff, 2.0) / (DCT_H * DCT_W)
    return (torch.sum(hvs_tiles.reshape(*dif.shape[:-2], DCT_H * DCT_W), dim=-1),
            weighted_mse / (DCT_H * DCT_W))


def hvs_hvsm_mse(images_a: torch.Tensor, images_b: torch.Tensor, batch: bool = False, masking_epsilon: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(images_a, np.ndarray):
        images_a = torch.tensor(images_a)
    if isinstance(images_b, np.ndarray):
        images_b = torch.tensor(images_b)

    hvs_tiles, hvsm_tiles = hvs_hvsm_mse_tiles(images_a, images_b, masking_epsilon=masking_epsilon)

    if batch or len(hvs_tiles.shape) < 2:
        return hvs_tiles.mean(dim=-1), hvsm_tiles.mean(dim=-1)
    else:
        return hvs_tiles.mean(dim=(0, -1)), hvsm_tiles.mean(dim=(0, -1))


def psnr_hvs_hvsm(images_a: torch.Tensor, images_b: torch.Tensor, batch: bool = False, masking_epsilon: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(images_a, np.ndarray):
        images_a = torch.tensor(images_a)
    if isinstance(images_b, np.ndarray):
        images_b = torch.tensor(images_b)

    hvs_tiles, hvsm_tiles = hvs_hvsm_mse_tiles(images_a, images_b, masking_epsilon=masking_epsilon)

    if batch or len(hvs_tiles.shape) < 2:
        return get_psnr(hvs_tiles.mean(dim=-1), 1.0), get_psnr(hvsm_tiles.mean(dim=-1), 1.0)
    else:
        return get_psnr(hvs_tiles.mean(dim=-1), 1.0).mean(dim=0), get_psnr(hvsm_tiles.mean(dim=-1), 1.0).mean(dim=0)
