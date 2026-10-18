import pytest


FLOAT_PRECISION_METRICS = {
    'psnr_hvs': 15.61038575899666,
    'psnr_hvsm': 15.966472675229104,
    'psnr_ha': 25.96774318769922,
    'psnr_hma': 26.467399309332656,
    'psnr': 18.04694023790249,
}


def torch_param(implementation):
    return pytest.param(
        implementation,
        id='torch',
        marks=pytest.mark.skipif(implementation is None, reason='PyTorch is not installed'),
    )
