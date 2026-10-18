"""Subprocess tests for the CLI seam: ``python -m psnr_hvsm``."""
import json
import subprocess
import sys
from pathlib import Path

from numpy.testing import assert_almost_equal

from conftest import FLOAT_PRECISION_METRICS

test_dir = Path(__file__).parent.absolute()
original = (test_dir / 'kodim01_c.png').as_posix()
distorted = (test_dir / 'kodim01_c_contrast.png').as_posix()

# current-release outputs for the default run (numpy backend) - the regression
# pin proving the default reports exactly the same numbers as before #16
default_values = {
    'psnr_hvs': 15.608883985024068,
    'psnr_hvsm': 15.964741058498255,
    'psnr_ha': 25.953298236404656,
    'psnr_hma': 26.46149718211291,
    'psnr': FLOAT_PRECISION_METRICS['psnr'],
}


def run_cli(*argv):
    """Run the CLI in a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, '-m', 'psnr_hvsm', *argv],
        capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def parse_last_stdout_line(stdout):
    # the package prints backend-availability notices on stdout, so the
    # document / metric line is the last line of the output
    return stdout.strip().splitlines()[-1]


def test_cli_default_json_records_8bit_precision():
    returncode, stdout, _ = run_cli('--json', original, distorted)

    assert returncode == 0
    document = json.loads(parse_last_stdout_line(stdout))
    assert document['precision'] == '8-bit'
    assert document['original'] == original
    assert document['distorted'] == distorted
    for key, value in default_values.items():
        assert_almost_equal(document[key], value, decimal=12)


def test_cli_float_precision_json_records_float():
    returncode, stdout, _ = run_cli('--json', '--precision', 'float', original, distorted)

    assert returncode == 0
    document = json.loads(parse_last_stdout_line(stdout))
    assert document['precision'] == 'float'
    # values pinned by the library-level #15 tests (exact rational derivation)
    for key in ('psnr_hvs', 'psnr_hvsm', 'psnr_ha', 'psnr_hma'):
        assert_almost_equal(document[key], FLOAT_PRECISION_METRICS[key], decimal=10)
    # plain PSNR does not depend on the BT.601 conversion, so it must match the
    # default run exactly. Compare against a default run of this environment
    # rather than the literal above: plain PSNR is unaffected by the conversion
    # but its last bits do follow the resolved NumPy version, so only two runs
    # that share an environment can be required to agree bit for bit.
    _, default_stdout, _ = run_cli('--json', original, distorted)
    assert document['psnr'] == json.loads(parse_last_stdout_line(default_stdout))['psnr']
    # the flag measurably changes every BT.601-dependent metric
    for key in ('psnr_hvs', 'psnr_hvsm', 'psnr_ha', 'psnr_hma'):
        assert document[key] != default_values[key]


def test_cli_rejects_precision_outside_choices():
    returncode, stdout, stderr = run_cli('--json', '--precision', '16-bit', original, distorted)

    # argparse rejects the value before any metric is computed
    assert returncode == 2
    assert 'precision' not in stdout


def test_cli_human_readable_output_unchanged():
    returncode, stdout, _ = run_cli(original, distorted)

    assert returncode == 0
    # byte-identical to the current release: same format, same numbers, and no
    # precision bookkeeping leaking into the human-readable line
    assert parse_last_stdout_line(stdout) == (
        'PSNR-HVS=15.608884, PSNR-HVS-M=15.964741, '
        'PSNR-HA=25.953298, PSNR-HMA=26.461497, PSNR=18.046940')


def test_cli_precision_agrees_with_library():
    from imageio.v2 import imread

    from psnr_hvsm.compute_all_metrics import compute_all_metrics

    org = imread(original)
    dst = imread(distorted)

    for precision in ('8-bit', 'float'):
        returncode, stdout, _ = run_cli('--json', '--precision', precision, original, distorted)

        assert returncode == 0
        document = json.loads(parse_last_stdout_line(stdout))
        library = compute_all_metrics(org, dst, precision=precision)
        for key, library_value in zip(('psnr_hvs', 'psnr_hvsm', 'psnr_ha', 'psnr_hma', 'psnr'), library):
            # same code path on the same backend: exact agreement required
            assert document[key] == library_value
