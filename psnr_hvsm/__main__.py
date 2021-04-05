import argparse

import numpy as np

from . import psnr_ha_hma, psnr_ha_hma_color, psnr_hvs_hvsm
from .bt601 import bt601ycbcr, bt601ypbpr
from .psnr import psnr

try:
    from imageio import imread
except ImportError as e:
    print('imageio not found. Please install psnr_hvsm with the [command_line] extra:')
    print('$> pip install psnr_hvsm[command_line]')
    raise e


parser = argparse.ArgumentParser(
    description='Compute the PSNR-HVS and PSNR-HVS-M metric between two images.')
parser.add_argument('original', type=str, help='The original image')
parser.add_argument('distorted', type=str, help='The distorted image')
parser.add_argument('--json', action='store_true', help='Output in JSON')
parser.add_argument('--hma-luma', action='store_true',
                    help='Force calculation of PSNR-HA and PSNR-HMA using only the luma channel even if the input is in color.')
parser.add_argument('--max-value', type=float, default=None, help='Maximum pixel intensity value (if image is not 8bpc)')

args = parser.parse_args()

org = imread(args.original)
dst = imread(args.distorted)

if len(org.shape) > 2:
    # discard alpha
    if org.shape[2] > 3:
        org = org[:, :, :3]
    if dst.shape[2] > 3:
        dst = dst[:, :, :3]

    # color processing
    if org.dtype == np.uint8:
        # use BT.601 YCbCr
        xy, xcb, xcr = bt601ycbcr(org)
        yy, ycb, ycr = bt601ycbcr(dst)
        org = org.astype(float) / 255
        dst = dst.astype(float) / 255
    else:
        if args.max_value is None:
            raise Exception('Please provide --max-value for image files that are >8bpc.')
        org = org.astype(float) / args.max_value
        dst = dst.astype(float) / args.max_value
        # assuming analog values (we don't know the ITU rec. to use)
        xy, xcb, xcr = bt601ypbpr(org)
        yy, ycb, ycr = bt601ypbpr(dst)

    if not args.hma_luma:
        psnr_ha, psnr_hma = psnr_ha_hma_color(xy, xcb, xcr, yy, ycb, ycr)
    else:
        psnr_ha, psnr_hma = psnr_ha_hma(xy, yy)

else:
    if org.dtype == np.uint8:
        xy = org.astype(float) / 255
        yy = dst.astype(float) / 255
    else:
        if args.max_value is None:
            raise Exception('Please provide --max-value for image files that are >8bpc.')
        xy = org.astype(float) / args.max_value
        yy = dst.astype(float) / args.max_value

    psnr_ha, psnr_hma = psnr_ha_hma(xy, yy)

psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(xy, yy)
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
            'psnr': psnr_mse
        }
    ))
else:
    print('PSNR-HVS=%f, PSNR-HVS-M=%f, PSNR-HA=%f, PSNR-HMA=%f, PSNR=%f' % (psnr_hvs, psnr_hvsm, psnr_ha, psnr_hma, psnr_mse))
