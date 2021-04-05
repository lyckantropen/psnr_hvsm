"""Compute the metrics for the TID2013 dataset."""

import argparse
from pathlib import Path

from .compute_all_metrics import compute_all_metrics

try:
    from imageio import imread
except ImportError as e:
    print('imageio not found. Please install psnr_hvsm with the [command_line] extra:')
    print('$> pip install psnr_hvsm[command_line]')
    raise e


parser = argparse.ArgumentParser(description='Compute metrics for the TID2013 dataset')
parser.add_argument('tid2013_path', type=str, help='Path to the TID2013 dataset root')
parser.add_argument('results', type=str, help='Path to a folder where to store the results')

args = parser.parse_args()

refs = sorted(list((Path(args.tid2013_path) / 'reference_images').glob('*.bmp')))

psnr_hvs = []
psnr_hvsm = []
psnr_ha = []
psnr_hma = []
psnr = []

for ref in refs:
    ref_img = imread(ref.as_posix())
    for d in range(1, 25):
        for i in range(1, 6):
            tst_path = (Path(args.tid2013_path) / 'distorted_images') / f'{ref.stem}_{d:02}_{i}.bmp'
            tst_img = imread(tst_path.as_posix())

            mtr = compute_all_metrics(ref_img, tst_img)
            [x.append(y) for x, y in zip([psnr_hvs, psnr_hvsm, psnr_ha, psnr_hma, psnr], mtr)]

(Path(args.results) / 'psnr_hvs.txt').write_text('\n'.join([str(p) for p in psnr_hvs]))
(Path(args.results) / 'psnr_hvsm.txt').write_text('\n'.join([str(p) for p in psnr_hvsm]))
(Path(args.results) / 'psnr_ha.txt').write_text('\n'.join([str(p) for p in psnr_ha]))
(Path(args.results) / 'psnr_hma.txt').write_text('\n'.join([str(p) for p in psnr_hma]))
(Path(args.results) / 'psnr.txt').write_text('\n'.join([str(p) for p in psnr]))
