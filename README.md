![cibuildwheel](https://github.com/lyckantropen/psnr_hvsm/actions/workflows/build_wheels.yml/badge.svg)

# psnr_hvsm

Accelerated Python package for computing the PSNR-HVS-M image metric.

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
