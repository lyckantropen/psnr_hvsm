import os
from typing import Callable, Dict

from .numpy import bt601ycbcr as bt601ycbcr_np
from .numpy import bt601ypbpr as bt601ypbpr_np
from .numpy import ha_hma_mse as ha_hma_mse_np
from .numpy import hvs_hvsm_mse as hvs_hvsm_mse_np
from .numpy import hvs_hvsm_mse_tiles as hvs_hvsm_mse_tiles_np
from .numpy import psnr as psnr_np
from .numpy import psnr_ha_hma as psnr_ha_hma_np
from .numpy import psnr_ha_hma_color as psnr_ha_hma_color_np
from .numpy import psnr_hvs_hvsm as psnr_hvs_hvsm_np

BACKENDS: Dict[str, Dict[str, Callable]] = {}
BACKENDS['numpy'] = {
    'psnr_hvs_hvsm': psnr_hvs_hvsm_np,
    'hvs_hvsm_mse': hvs_hvsm_mse_np,
    'hvs_hvsm_mse_tiles': hvs_hvsm_mse_tiles_np,
    'psnr': psnr_np,
    'bt601ycbcr': bt601ycbcr_np,
    'bt601ypbpr': bt601ypbpr_np,
    'psnr_ha_hma': psnr_ha_hma_np,
    'psnr_ha_hma_color': psnr_ha_hma_color_np,
    'ha_hma_mse': ha_hma_mse_np,
}

try:
    from .torch import bt601ycbcr as bt601ycbcr_pt
    from .torch import bt601ypbpr as bt601ypbpr_pt
    from .torch import ha_hma_mse as ha_hma_mse_pt
    from .torch import hvs_hvsm_mse as hvs_hvsm_mse_pt
    from .torch import hvs_hvsm_mse_tiles as hvs_hvsm_mse_tiles_pt
    from .torch import psnr as psnr_pt
    from .torch import psnr_ha_hma as psnr_ha_hma_pt
    from .torch import psnr_ha_hma_color as psnr_ha_hma_color_pt
    from .torch import psnr_hvs_hvsm as psnr_hvs_hvsm_pt
    BACKENDS['torch'] = {
        'psnr_hvs_hvsm': psnr_hvs_hvsm_pt,
        'hvs_hvsm_mse': hvs_hvsm_mse_pt,
        'hvs_hvsm_mse_tiles': hvs_hvsm_mse_tiles_pt,
        'psnr': psnr_pt,
        'bt601ycbcr': bt601ycbcr_pt,
        'bt601ypbpr': bt601ypbpr_pt,
        'psnr_ha_hma': psnr_ha_hma_pt,
        'psnr_ha_hma_color': psnr_ha_hma_color_pt,
        'ha_hma_mse': ha_hma_mse_pt,
    }
except ImportError:
    print('PyTorch not installed, if you wish to use PyTorch, install as psnr_hvsm[torch]:')
    print('$> pip install psnr_hvsm[torch]')
    pass

try:
    from ._psnr_hvsm import hvs_hvsm_mse as hvs_hvsm_mse_cpp
    from ._psnr_hvsm import hvs_hvsm_mse_tiles as hvs_hvsm_mse_tiles_cpp
    from ._psnr_hvsm import psnr_hvs_hvsm as psnr_hvs_hvsm_cpp
    BACKENDS['cpp'] = {
        'psnr_hvs_hvsm': psnr_hvs_hvsm_cpp,
        'hvs_hvsm_mse': hvs_hvsm_mse_cpp,
        'hvs_hvsm_mse_tiles': hvs_hvsm_mse_tiles_cpp,
        'psnr': psnr_np,
        'bt601ycbcr': bt601ycbcr_np,
        'bt601ypbpr': bt601ypbpr_np,
        'psnr_ha_hma': psnr_ha_hma_np,
        'psnr_ha_hma_color': psnr_ha_hma_color_np,
        'ha_hma_mse': ha_hma_mse_np,
    }
except ImportError:
    print('C++ version not available.')

if os.environ.get('PSNR_HVSM_BACKEND', None) is not None:
    backend = os.environ['PSNR_HVSM_BACKEND']
else:
    backend = 'numpy'

psnr_hvs_hvsm = BACKENDS[backend]['psnr_hvs_hvsm']
hvs_hvsm_mse = BACKENDS[backend]['hvs_hvsm_mse']
hvs_hvsm_mse_tiles = BACKENDS[backend]['hvs_hvsm_mse_tiles']
psnr = BACKENDS[backend]['psnr']
bt601ycbcr = BACKENDS[backend]['bt601ycbcr']
bt601ypbpr = BACKENDS[backend]['bt601ypbpr']
psnr_ha_hma = BACKENDS[backend]['psnr_ha_hma']
psnr_ha_hma_color = BACKENDS[backend]['psnr_ha_hma_color']
ha_hma_mse = BACKENDS[backend]['ha_hma_mse']
