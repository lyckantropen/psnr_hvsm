"""Entry point for running the package as a module. Computes the metrics from the command line."""

import argparse

from .compute_all_metrics import compute_all_metrics

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


psnr_hvs, psnr_hvsm, psnr_ha, psnr_hma, psnr_mse = compute_all_metrics(org, dst, args.max_value, args.hma_luma)

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
