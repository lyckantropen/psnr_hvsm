![cibuildwheel](https://github.com/lyckantropen/psnr_hvsm/actions/workflows/build_wheels.yml/badge.svg)

# psnr_hvsm

Accelerated Python package for computing the PSNR-HVS-M image metric.

This is an implementation of the PSNR-HVS and PSNR-HVS-M metrics developed by
[Nikolay Ponomarenko](www.ponomarenko.info/psnrhvsm).

## Bibliography

* [Egiazarian, Karen, et al. "New full-reference quality metrics based on HVS." Proceedings of the Second International Workshop on Video Processing and Quality Metrics. Vol. 4. 2006.](https://www.researchgate.net/profile/Vladimir_Lukin2/publication/251229783_A_NEW_FULL-REFERENCE_QUALITY_METRICS_BASED_ON_HVS/links/0046351f669a9c1869000000.pdf)
* [Ponomarenko, Nikolay, et al. "On between-coefficient contrast masking of DCT basis functions." Proceedings of the third international workshop on video processing and quality metrics. Vol. 4. 2007.](https://www.researchgate.net/profile/Vladimir-Lukin-4/publication/242309240_On_between-coefficient_contrast_masking_of_DCT_basis_functions/links/0c96052442be7c3176000000/On-between-coefficient-contrast-masking-of-DCT-basis-functions.pdf)

## Building

### Dependencies

`psnr_hvsm` has several dependencies:

* [FFTW3 >= 3.3.9](http://www.fftw.org/)
* [pybind11](https://github.com/pybind/pybind11)
* [xtensor](https://github.com/xtensor-stack/xtensor)
* [xtensor-fftw](https://github.com/xtensor-stack/xtensor-fftw)
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
