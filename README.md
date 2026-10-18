# PSNR-HVS-M, PSNR-HA and PSNR-HMA metrics for NumPy and PyTorch

[![cibuildwheel](https://github.com/lyckantropen/psnr_hvsm/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/lyckantropen/psnr_hvsm/actions)
[![python_versions](https://img.shields.io/pypi/pyversions/psnr_hvsm)](https://pypi.org/project/psnr-hvsm/)
[![pypi](https://img.shields.io/pypi/v/psnr_hvsm)](https://pypi.org/project/psnr-hvsm/)
[![license](https://img.shields.io/github/license/lyckantropen/psnr_hvsm)](https://github.com/lyckantropen/psnr_hvsm/blob/main/LICENSE)

Accelerated Python package for computing several image metrics based on human
perception with backends in NumPy, PyTorch and C++.

This is an implementation of the PSNR-HVS, PSNR-HVS-M, PSNR-HA and PSNR-HMA
metrics developed by
[Nikolay Ponomarenko](http://www.ponomarenko.info/psnrhvsm.htm).

The values produced by this library have been cross-checked against the results
within the
[TID2013 dataset](https://www.sciencedirect.com/science/article/pii/S0923596514001490).
(See the folder [`tid2013_results`](tid2013_results).) The only difference is
that this library follows the common convention that PSNR for identical signals
equals 100.0.

_A miniscule discrepancy for PSNR-HMA (<0.01dB on average) is under
investigation._

## Bibliography

- [Egiazarian, Karen, et al. "New full-reference quality metrics based on HVS." Proceedings of the Second International Workshop on Video Processing and Quality Metrics. Vol. 4. 2006.](https://www.researchgate.net/profile/Vladimir_Lukin2/publication/251229783_A_NEW_FULL-REFERENCE_QUALITY_METRICS_BASED_ON_HVS/links/0046351f669a9c1869000000.pdf)
- [Ponomarenko, Nikolay, et al. "On between-coefficient contrast masking of DCT basis functions." Proceedings of the third international workshop on video processing and quality metrics. Vol. 4. 2007.](https://www.researchgate.net/profile/Vladimir-Lukin-4/publication/242309240_On_between-coefficient_contrast_masking_of_DCT_basis_functions/links/0c96052442be7c3176000000/On-between-coefficient-contrast-masking-of-DCT-basis-functions.pdf)
- [Ponomarenko, Nikolay, et al. "Modified image visual quality metrics for contrast change and mean shift accounting." 2011 11th International Conference The Experience of Designing and Application of CAD Systems in Microelectronics (CADSM). IEEE, 2011.](https://ponomarenko.info/papers/psnrhma.pdf)

## Installation

`psnr_hvsm` supports Python 3.10-3.15. Packages are distributed on PyPi. Be sure
to have an up-to-date pip to be able to install the correct packages on Linux:

```bash
python -m pip install --upgrade pip
```

```bash
pip install psnr_hvsm
```

With PyTorch support:

```bash
pip install psnr_hvsm[torch]
```

### NumPy and PyTorch support policy

The NumPy requirement carries no upper bound. The compiled extension is built
against NumPy 2 headers with a fixed backward-compatible ABI target, so a
single wheel serves both NumPy 1.x and NumPy 2.x runtimes. Installing the
package never forces a downgrade from NumPy 2.x, and its runtime floor is
`numpy>=1.19`. Note that NumPy 1.x itself
is only installable on Python 3.10-3.12: the last 1.x release ships wheels
only that far, so on Python 3.13 and newer the runtime resolves NumPy 2.x.

PyPI publishes PyTorch wheels for Python 3.10-3.14 only, so the `torch` extra
is unavailable on 3.15 and the PyTorch backend cannot be installed there.

### MSVC runtime on Windows

Linux manylinux wheels bundle `libstdc++` via auditwheel. The Windows wheels do
not bundle the MSVC C++ runtime: CMake installs only the FFTW DLLs into the
wheel. On CPython 3.10-3.14 the package
therefore declares `msvc-runtime` for Windows, which pip installs into the
environment along with the package, so no system-wide Visual C++ runtime is
required. The MSVC or clang toolchain supplies the CRT when the extension is
compiled, which is why it is not a build requirement.

On CPython 3.15 the requirement is not declared. `msvc-runtime` publishes its
wheels per CPython ABI and has no cp315 wheel, so asking for it there would
leave the newest supported interpreter uninstallable. A CPython install from
python.org brings the C runtime it was built with (its installer updates the
C Runtime Library on the machine when one is needed), so 3.15 works without
anything extra. If an import nevertheless fails with a missing
`vcruntime140.dll` or `msvcp140.dll` - a possibility with the minimal layouts
such as the embeddable package, MSYS2 or NuGet-based installs - install the
Microsoft Visual C++ 2015-2022 redistributable for your architecture, or use
CPython 3.10-3.14, where the requirement is declared.

## Usage

### Command line

Command line support is an extra that pulls `imageio`:

```bash
pip install psnr_hvsm[command_line]
```

```bash
python -m psnr_hvsm original.png distorted.png
```

#### Choosing a backend

The backend can be set by setting the `PSNR_HVSM_BACKEND` environment variable.
Valid backends are:

- `numpy` - pure NumPy using `scipy.fft` for DCT
- `cpp` - C++ using FFTW3
- `torch` - PyTorch; install as `psnr_hvsm[torch]` to install PyTorch as well

```bash
export PSNR_HVSM_BACKEND=torch
python -m psnr_hvsm original.png distorted.png
```

The default device for PyTorch is `cuda` but it can be changed by setting the
`PSNR_HVSM_TORCH_DEVICE` environment variable.

### As a library

The function `psnr_hvs_hvsm` accepts images as single-channel floating-point
NumPy arrays. The images need to be normalised, i.e. the values need to be in
the range `[0,1]`. This can be achieved by converting the image to `float` and
dividing by the maximum value given the bit depth. For 8 bits per component this
is 255.

The images must be padded to a multiple of 8 in each dimension.

```python
from imageio import imread
from psnr_hvsm import psnr_hvs_hvsm, bt601ycbcr

image1 = imread('tests/baboon.png').astype(float) / 255
image2 = imread('tests/baboon_msk.png').astype(float) / 255

image1_y, *_ = bt601ycbcr(image1)
image2_y, *_ = bt601ycbcr(image2)

psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(image1, image2)

print(psnr_hvs, psnr_hvsm)
```

```bash
34.427054505764424 51.64722121999962
```

If you need to measure PSNR-HVS and PSNR-HVS-M on an RGB image, you need to
convert it to an YUV colorspace and pass in only the luma component.

### BT.601 conversion precision

When 8-bit RGB input is converted to the BT.601 YCbCr colorspace, the luma
and both chroma components are by default rounded to 8-bit code values. This
quantisation matches the 8-bit YCbCr pipeline of the reference implementation
and is what reproduces the published TID2013 reference values this library is
cross-checked against, which is why it remains the default. Every metric
output computed with the default is bit-identical to previous releases.

If you are porting the metrics and want to compare against a non-quantised
reference, pass `precision='float'` to compute the same BT.601 matrix
arithmetic without the rounding step:

```python
from psnr_hvsm import bt601ycbcr
from psnr_hvsm.compute_all_metrics import compute_all_metrics

image1_y, image1_cb, image1_cr = bt601ycbcr(image1_uint8, precision='float')

psnr_hvs, psnr_hvsm, psnr_ha, psnr_hma, psnr = compute_all_metrics(
    image1_uint8, image2_uint8, precision='float')
```

The `precision` option (`'8-bit'` default, `'float'`) is accepted by
`bt601ycbcr` in both the NumPy and PyTorch backends with identical
semantics, and by the `compute_all_metrics` entry point, which threads it
through every metric that depends on the quantised 8-bit BT.601 YCbCr
conversion. It applies only when that conversion runs. For grayscale input or
non-uint8 color input, it has no effect.

On the command line the same choice is exposed as
`--precision {8-bit,float}` on `python -m psnr_hvsm`, and the JSON output
(`--json`) records the requested precision, so batch results are
self-describing. The requested precision is applied wherever the quantised
conversion runs; for grayscale or non-uint8 color input, the field has no
effect.

### PyTorch support

The PyTorch backend can be used for use with gradient descent algorithms and
computation on GPUs. In order to use the PyTorch backend, either install the
package with the `torch` extra:

```bash
pip install psnr_hvsm[torch]
```

If your PyTorch installation was manual, you need
[`torch-dct`](https://github.com/zh217/torch-dct) in order to use the PyTorch
backend:

```bash
pip install "torch-dct>=0.1.6"
```

An important distinction is that the functions that expect 3-channel input now
expect `(...,C,H,W)` format in the PyTorch implementation. The PyTorch backend
can be enabled by importing it directly from `psnr_hvsm.torch`:

```python
import torch
from imageio import imread
from psnr_hvsm.torch import psnr_hvs_hvsm, bt601ycbcr

image1 = imread('tests/baboon.png').astype(float) / 255
image2 = imread('tests/baboon_msk.png').astype(float) / 255

image1 = torch.tensor(image1, device='cuda').moveaxis(-1, -3)  # convert to (N,C,H,W) format
image2 = torch.tensor(image2, device='cuda').moveaxis(-1, -3)  # convert to (N,C,H,W) format

image1_y, *_ = bt601ycbcr(image1)
image2_y, *_ = bt601ycbcr(image2)

psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(image1_y, image2_y)
```

Alternatively, set the `PSNR_HVSM_BACKEND` environment variable to `torch`:

```python
import os
os.environ['PSNR_HVSM_BACKEND'] = 'torch'

from psnr_hvsm import psnr_hvs_hvsm
# rest of code
# ...
```

#### Note on gradients

Some operations in the calculation of the HVS-M MSE lead to problems with the
gradient and therefore a parameter called `masking_epsilon` (defaulting to 0.0)
has been added to the PyTorch versions of `hvs_hvsm_mse_tiles`, `hvs_hvsm_mse`
and `psnr_hvs_hvsm`. Set it to a small value (determined by your own
experimentation) if you need the result of HVS-M for gradient descent.

### Computing metrics for the TID2013 dataset

If you have a copy of the
[TID2013 dataset](https://www.sciencedirect.com/science/article/pii/S0923596514001490),
you can re-verify the metrics for yourself:

```bash
python -m psnr_hvsm.tid2013_metrics D:\tid2013\ .\tid2013_results\
```

### Other exported functions

- `hvs_hvsm_mse_tiles` - compute HVS and HVS-M scores on all 8x8 tiles in the
  images, returns an array of numbers
- `hvs_hvsm_mse` - compute average HVS and HVS-M scores

## Building

### Dependencies

`psnr_hvsm` has several dependencies:

- [FFTW3 >= 3.3.9](http://www.fftw.org/)
- [pybind11](https://github.com/pybind/pybind11)
- [xtensor](https://github.com/xtensor-stack/xtensor)
- [xtensor-python](https://github.com/xtensor-stack/xtensor-python)

FFTW3 is always fetched by CMake at the pinned release listed in
`psnr_hvsm/CMakeLists.txt`. The remaining, header-only dependencies are
installed at pinned, mutually compatible versions when they are not already
installed, so a C++ compiler, a current CMake and pip or uv are enough to build
from source:

```bash
pip install --upgrade -r requirements-build.txt
pip wheel --no-deps -w dist .
```

If you prefer a `conda` environment, create the provided one and CMake will
discover the dependencies from it instead of fetching them:

```bash
conda env create -f psnr_hvsm-dev.yml
```

### Development mode

To install in development mode:

```bash
pip install --upgrade -r requirements.txt
```

### Creating Python wheel

```bash
pip install --upgrade -r requirements-build.txt
python -m build --wheel
```

### Verifying a clean source build

To build from source in a scratch virtualenv that has no conda environment for
CMake to discover and to run the test suite against the result:

```bash
./clean-build-check.sh
```

### Running tests on different versions of Python using tox

```bash
pip install --upgrade -r requirements-tox.txt
tox --parallel auto
```
