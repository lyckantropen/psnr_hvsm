import torch


def get_psnr(s: torch.Tensor, mv: float) -> torch.Tensor:
    return torch.where(s != 0.0, 10.0 * torch.log10(torch.div(mv**2, s)), torch.full_like(s, 100.0)).clip(0, 100.0)


def psnr(a: torch.Tensor, b: torch.Tensor, mv: float = 1.0) -> torch.Tensor:
    return get_psnr(torch.pow(a - b, 2.0).mean(), mv)
