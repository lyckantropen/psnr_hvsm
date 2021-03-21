#include "dct.h"

#include <xtensor-fftw/basic.hpp>
#include <xtensor-fftw/helper.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xfunctor_view.hpp>

using xt::placeholders::_;

// DCT-II performed using FFT
xt::xtensor<double, 1> dct(const xt::xtensor<double, 1> &x)
{
  xt::xarray<double> f = xt::zeros<double>({x.shape(0) * 4});
  auto a = xt::view(x, xt::range(_, _, -1));
  auto b = xt::view(x, xt::all());
  auto c = xt::concatenate(std::make_tuple(a, b));
  xt::view(f, xt::range(1, _, 2)) = c;
  auto fft = xt::real(xt::fftw::rfft(f));
  xt::view(fft, xt::range(1, _, 2)) *= -1.0;

  // normalization
  fft(0) *= std::sqrt(1.0/(4.*x.size()));
  xt::view(fft, xt::range(1, _)) *= std::sqrt(1.0/(2.*x.size()));

  return xt::view(fft, xt::range(_, x.shape(0)));
}

// 2D DCT-II
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
