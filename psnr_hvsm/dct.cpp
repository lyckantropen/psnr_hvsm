#include "dct.h"

#include <fftw3.h>
#include <xtensor/xview.hpp>

namespace psnr_hvsm
{
  using xt::placeholders::_;

  xt::xtensor<double, 1> dct(xt::xtensor<double, 1> x, bool norm)
  {
    fftw_plan plan = fftw_plan_r2r_1d(x.shape(0), x.data(), x.data(), FFTW_REDFT10, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    fftw_execute(plan);

    // normalization
    if (norm)
    {
      x(0) *= std::sqrt(1.0 / (4. * x.size()));
      xt::view(x, xt::range(1, _)) *= std::sqrt(1.0 / (2. * x.size()));
    }

    return x;
  }

  xt::xtensor<double, 2> dct2(xt::xtensor<double, 2> x, bool norm)
  {
    fftw_plan plan = fftw_plan_r2r_2d(x.shape(0), x.shape(1), x.data(), x.data(), FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    fftw_execute(plan);

    // normalization
    if (norm)
    {
      x /= std::sqrt(x.shape(0) * x.shape(1));
      x(0, 0) /= 4.;
      xt::view(xt::col(x, 0), xt::range(1, _)) /= 2 * std::sqrt(2.);
      xt::view(xt::row(x, 0), xt::range(1, _)) /= 2 * std::sqrt(2.);
      xt::view(x, xt::range(1, _), xt::range(1, _)) /= 2;
    }
    return x;
  }
}