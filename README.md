[![cibuildwheel](https://github.com/lyckantropen/psnr_hvsm/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/lyckantropen/psnr_hvsm/actions)
[![python_versions](https://img.shields.io/pypi/pyversions/psnr_hvsm)](https://pypi.org/project/psnr-hvsm/)
[![pypi](https://img.shields.io/pypi/v/psnr_hvsm)](https://pypi.org/project/psnr-hvsm/)
[![license](https://img.shields.io/github/license/lyckantropen/psnr_hvsm)](https://github.com/lyckantropen/psnr_hvsm/blob/main/LICENSE)

# psnr_hvsm

Accelerated Python package for computing several image metrics based on human
perception.

This is an implementation of the PSNR-HVS, PSNR-HVS-M, PSNR-HA and PSNR-HMA
metrics developed by [Nikolay
Ponomarenko](http://www.ponomarenko.info/psnrhvsm).

The values produced by this library have been cross-checked against the results
within the [TID2013
dataset](https://www.sciencedirect.com/science/article/pii/S0923596514001490).
(See the folder [`tid2013_results`](tid2013_results).) The only difference is
that this library follows the common convention that PSNR for identical signals
equals 100.0.

## Bibliography

* [Egiazarian, Karen, et al. "New full-reference quality metrics based on HVS." Proceedings of the Second International Workshop on Video Processing and Quality Metrics. Vol. 4. 2006.](https://www.researchgate.net/profile/Vladimir_Lukin2/publication/251229783_A_NEW_FULL-REFERENCE_QUALITY_METRICS_BASED_ON_HVS/links/0046351f669a9c1869000000.pdf)
* [Ponomarenko, Nikolay, et al. "On between-coefficient contrast masking of DCT basis functions." Proceedings of the third international workshop on video processing and quality metrics. Vol. 4. 2007.](https://www.researchgate.net/profile/Vladimir-Lukin-4/publication/242309240_On_between-coefficient_contrast_masking_of_DCT_basis_functions/links/0c96052442be7c3176000000/On-between-coefficient-contrast-masking-of-DCT-basis-functions.pdf)
* [Ponomarenko, Nikolay, et al. "Modified image visual quality metrics for contrast change and mean shift accounting." 2011 11th International Conference The Experience of Designing and Application of CAD Systems in Microelectronics (CADSM). IEEE, 2011.](https://www.researchgate.net/profile/Marco_Carli/publication/261094981_Color_image_database_TID2013_Peculiarities_and_preliminary_results/links/0deec536c86764dad6000000/Color-image-database-TID2013-Peculiarities-and-preliminary-results.pdf)

## Installation

`psnr_hvsm` supports Python 3.6-3.9. Packages are distributed on PyPi. Be sure
to have an up-to-date pip to be able to install the correct packages on Linux:

```bash
python -m pip install --upgrade pip
```

```bash
pip install psnr_hvsm
```

## Usage

### Command line

Command line support is an extra that pulls `opencv-python-headless`:

```bash
pip install psnr_hvsm[command_line]
```

```bash
python -m psnr_hvsm original.png distorted.png
```

### As a library

The function `psnr_hvs_hvsm` accepts images as single-channel floating-point
NumPy arrays. The images need to be normalised, i.e. the values need to be in
the range `[0,1]`. This can be achieved by converting the image to `float` and
dividing by the maximum value given the bit depth. For 8 bits per component
this is 255.

The images must be padded to a multiple of 8 in each dimension.

```python
from imageio import imread
from psnr_hvsm import psnr_hvs_hvsm

image1 = imread('tests/baboon.png').astype(float) / 255
image2 = imread('tests/baboon_msk.png').astype(float) / 255

psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(image1, image2)

print(psnr_hvs, psnr_hvsm)
```

```bash
34.427054505764424 51.64722121999962
```

If you need to measure PSNR-HVS and PSNR-HVS-M on an RGB image, you need to
convert it to an YUV colorspace and pass in only the luma component.

### Other exported functions

* `psnr_hvs` - returns only the PSNR-HVS value
* `psnr_hvsm` - returns only the PSNR-HVS-M value
* `hvs_mse_tile` - compute the weighted MSE of two 8x8 tiles
* `hvsm_mse_tile` - compute the weighted MSE with masking correction of two 8x8 tiles
* `hvs_mse` - compute HVS scores on all 8x8 tiles in the images, returns an array of numbers
* `hvsm_mse` - compute HVS-M scores on all 8x8 tiles in the images, returns an array of numbers

## Building

### Dependencies

`psnr_hvsm` has several dependencies:

* [FFTW3 >= 3.3.9](http://www.fftw.org/)
* [pybind11](https://github.com/pybind/pybind11)
* [xtensor](https://github.com/xtensor-stack/xtensor)
* [xtensor-python](https://github.com/xtensor-stack/xtensor-python)

All of the above can be automatically resolved by running `deps.ps1`, which is
a cross-platform PowerShell script (i.e. it can also be run under Linux if you
have PowerShell installed).

### Development mode

To install in development mode:

```bash
pip install --upgrade -r requirements.txt
```

### Creating Python wheel

```bash
pip install --upgrade -r requirements-build.txt
python setup.py bdist_wheel
```

### Running tests on different versions of Python using tox

```bash
pip install --upgrade -r requirements-tox.txt
tox --parallel auto
```
