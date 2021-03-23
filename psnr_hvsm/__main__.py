import argparse

import numpy as np

from . import psnr_hvs_hvsm
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

org = cv2.imread(args.original, cv2.IMREAD_UNCHANGED)
dst = cv2.imread(args.distorted, cv2.IMREAD_UNCHANGED)

if org.dtype == np.uint8:
    if len(org.shape) > 2:
        org = cv2.cvtColor(org, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YCrCb)[:, :, 0]
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
        org = org[:, :, 0] + 2*org[:, :, 1] + org[:, :, 2]
        dst = dst[:, :, 0] + 2*dst[:, :, 1] + dst[:, :, 2]


psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(org, dst)
psnr_mse = psnr(org, dst)  # add PSNR for good measure

if args.json:
    import json
    print(json.dumps(
        {
            'original': args.original,
            'distorted': args.distorted,
            'psnr_hvs': psnr_hvs,
            'psnr_hvsm': psnr_hvsm,
            'psnr': psnr_mse
        }
    ))
else:
    print('PSNR-HVS=%f, PSNR-HVS-M=%f, PSNR=%f' % (psnr_hvs, psnr_hvsm, psnr_mse))
