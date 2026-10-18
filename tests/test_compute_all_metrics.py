import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from imageio.v2 import imread
from numpy.testing import assert_almost_equal

from conftest import FLOAT_PRECISION_METRICS
from psnr_hvsm.compute_all_metrics import compute_all_metrics

test_dir = Path(__file__).parent.absolute()
org = imread((test_dir / 'kodim01_c.png').as_posix())
tst = imread((test_dir / 'kodim01_c_contrast.png').as_posix())


def test_compute_all_metrics_default_precision():
    metrics = compute_all_metrics(org, tst)

    # current-release values (8-bit rounded BT.601, numpy backend) - the regression
    # guard proving the default precision did not move any metric output
    assert_almost_equal(metrics[0], 15.608883985024068, decimal=12)
    assert_almost_equal(metrics[1], 15.964741058498255, decimal=12)
    assert_almost_equal(metrics[2], 25.953298236404656, decimal=12)
    assert_almost_equal(metrics[3], 26.46149718211291, decimal=12)
    assert_almost_equal(metrics[4], 18.04694023790249, decimal=12)


def test_compute_all_metrics_float_precision():
    metrics = compute_all_metrics(org, tst, precision='float')

    # expected values derived with an exact rational BT.601 conversion (no rounding)
    # feeding the numpy metric functions, independently of the implementation
    for metric, name in zip(metrics[:4], ('psnr_hvs', 'psnr_hvsm', 'psnr_ha', 'psnr_hma')):
        assert_almost_equal(metric, FLOAT_PRECISION_METRICS[name], decimal=10)
    # plain PSNR does not depend on the BT.601 conversion
    assert_almost_equal(metrics[4], FLOAT_PRECISION_METRICS['psnr'], decimal=12)


def test_compute_all_metrics_precision_divergence():
    # the rounding step measurably moves every BT.601-dependent metric (issue #9)
    default = compute_all_metrics(org, tst)
    flt = compute_all_metrics(org, tst, precision='float')

    for default_metric, float_metric in zip(default[:4], flt[:4]):
        assert default_metric != float_metric
    assert default[4] == flt[4]


def test_compute_all_metrics_invalid_precision():
    with pytest.raises(ValueError):
        compute_all_metrics(org, tst, precision='16-bit')


def _compute_in_torch_backend(precision):
    # mirrors the CLI's torch path: uint8 (H,W,C) images wrapped to (C,H,W) tensors
    script = (
        'import json, sys\n'
        'import torch\n'
        'from imageio.v2 import imread\n'
        'from psnr_hvsm.compute_all_metrics import compute_all_metrics\n'
        'org = imread(sys.argv[1])\n'
        'dst = imread(sys.argv[2])\n'
        'org = torch.tensor(org, device="cpu").moveaxis(-1, -3)\n'
        'dst = torch.tensor(dst, device="cpu").moveaxis(-1, -3)\n'
        'print(json.dumps(compute_all_metrics(org, dst, precision=sys.argv[3])))\n'
    )
    env = os.environ.copy()
    env['PSNR_HVSM_BACKEND'] = 'torch'
    env['PSNR_HVSM_TORCH_DEVICE'] = 'cpu'
    result = subprocess.run(
        [sys.executable, '-c', script,
         (test_dir / 'kodim01_c.png').as_posix(),
         (test_dir / 'kodim01_c_contrast.png').as_posix(),
         precision],
        capture_output=True, text=True, env=env, check=True)
    # the package prints backend-availability notices on stdout, so the JSON
    # document is the last line of the output
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_compute_all_metrics_torch_backend_agrees():
    pytest.importorskip('torch_dct')
    for precision in ('8-bit', 'float'):
        torch_metrics = _compute_in_torch_backend(precision)
        numpy_metrics = compute_all_metrics(org, tst, precision=precision)

        for torch_metric, numpy_metric in zip(torch_metrics, numpy_metrics):
            # the torch backend converts uint8 input in its default dtype
            # (float32), so agreement is asserted within float32 noise
            assert_almost_equal(torch_metric, numpy_metric, decimal=3)
