import argparse

import numpy as np

from . import psnr_ha_hma, psnr_hvs_hvsm
from .bt601 import bt601luma, bt601luma_norm
from .psnr import psnr

try:
    import cv2
except ImportError as e:
    print('OpenCV not found. Please install psnr_hvsm with the [command_line] extra:')
    print('$> pip install psnr_hvsm[command_line]')
    raise e


parser = argparse.ArgumentParser(description='Compute the PSNR-HVS and PSNR-HVS-M metric between two images.')
parser.add_argument('original', type=str, help='The original image')
parser.add_argument('distorted', type=str, help='The distorted image')
parser.add_argument('--json', action='store_true', help='Output in JSON')
parser.add_argument('--bits-per-component', type=int, default=None, help='Bits per component (if image is not 8bpc)')

args = parser.parse_args()

org = cv2.imread(args.original, cv2.IMREAD_ANYDEPTH)
dst = cv2.imread(args.distorted, cv2.IMREAD_ANYDEPTH)

if org.dtype == np.uint8:
    if len(org.shape) > 2:
        org = bt601luma(org)
        dst = bt601luma(dst)
    else:
        org = org.astype(float) / 255
        dst = dst.astype(float) / 255
else:
    if args.bits_per_component is None:
        raise Exception('Please provide --bits-per-component for image files that are >8bpc.')
    max_val = (1 << args.bits_per_component) - 1
    org = org.astype(float) / max_val
    dst = dst.astype(float) / max_val

    if len(org.shape) > 2:
        # simplified luma (since we don't really know which ITU rec. to use)
        org = bt601luma_norm(org)
        dst = bt601luma_norm(dst)


psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(org, dst)
psnr_ha, psnr_hma = psnr_ha_hma(org, dst)
psnr_mse = psnr(org, dst)  # add PSNR for good measure

if args.json:
    import json
    print(json.dumps(
        {
            'original': args.original,
            'distorted': args.distorted,
            'psnr_hvs': psnr_hvs,
            'psnr_hvsm': psnr_hvsm,
            'psnr_ha': psnr_ha,
            'psnr_hma': psnr_hma,
            'psnr_y': psnr_mse
        }
    ))
else:
    print('PSNR-HVS=%f, PSNR-HVS-M=%f, PSNR-HA=%f, PSNR-HMA=%f, PSNR-Y=%f' % (psnr_hvs, psnr_hvsm, psnr_ha, psnr_hma, psnr_mse))
