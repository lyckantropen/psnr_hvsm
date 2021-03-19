#include <xtensor/xtensor.hpp>
#include <pybind11/pybind11.h>

#include <xtensor-fftw/basic.hpp>  // rfft, irfft
#include <xtensor-fftw/helper.hpp> // rfftscale
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp> // xt::arange
#include <xtensor/xmath.hpp>    // xt::sin, cos
#include <complex>
#include <xtensor/xio.hpp>

#include <xtensor/xview.hpp>
#include <xtensor/xfunctor_view.hpp>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

using xt::placeholders::_;

xt::xtensor<double, 1> dct(const xt::xtensor<double, 1> &x)
{
  xt::xarray<double> f = xt::zeros<double>({x.shape(0) * 4});
  auto a = xt::view(x, xt::range(_, _, -1));
  auto b = xt::view(x, xt::all());
  auto c = xt::concatenate(std::make_tuple(a, b));
  xt::view(f, xt::range(1, _, 2)) = c;
  auto fft = xt::real(xt::fftw::rfft(f));
  xt::view(fft, xt::range(1, _, 2)) *= -1.0;
  return xt::view(fft, xt::range(_, x.shape(0)));
}

xt::xtensor<double, 2> dct2(const xt::xtensor<double, 2> &x)
{
  xt::xtensor<double, 2> y = x;
  for (size_t j = 0; j < x.shape(1); ++j)
  {
    xt::row(y, j) = dct(xt::row(y, j));
  }
  for (size_t i = 0; i < x.shape(0); ++i)
  {
    xt::col(y, i) = dct(xt::col(y, i));
  }
  return y;
}

PYBIND11_MODULE(_psnr_hvsm, m)
{
  xt::import_numpy();

  m.doc() = "PSNR-HVS-M";
  m.def("dct", [](xt::pytensor<double, 1> x) -> xt::pytensor<double, 1> {
    return xt::pytensor<double, 1>{dct(x)};
  });
  m.def("dct2", [](xt::pytensor<double, 2> x) -> xt::pytensor<double, 2> {
    return xt::pytensor<double, 2>{dct2(x)};
  });
}