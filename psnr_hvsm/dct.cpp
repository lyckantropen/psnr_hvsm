#include "dct.h"

#include <map>
#include <fftw3.h>
#include <xtensor/xview.hpp>

namespace psnr_hvsm
{
  using xt::placeholders::_;

  static std::map<size_t, fftw_plan> dct_plan;
  static std::map<std::pair<size_t,size_t>, fftw_plan> dct2_plan;

  xt::xtensor<double, 1> dct(xt::xtensor<double, 1> x, bool norm)
  {
    #ifdef PSNR_HVSM_OPENMP
    #pragma omp critical
    #endif
    {
      if(dct_plan.find(x.shape(0)) == std::end(dct_plan)) {
        dct_plan[x.shape(0)] = fftw_plan_r2r_1d(x.shape(0), x.data(), x.data(), FFTW_REDFT10, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
      }
    }
    fftw_execute_r2r(dct_plan[x.shape(0)], x.data(), x.data());

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
    #ifdef PSNR_HVSM_OPENMP
    #pragma omp critical
    #endif
    {
      if(dct2_plan.find(std::make_pair(x.shape(0), x.shape(1))) == std::end(dct2_plan)) {
        dct2_plan[std::make_pair(x.shape(0), x.shape(1))] = fftw_plan_r2r_2d(x.shape(0), x.shape(1), x.data(), x.data(), FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
      }
    }
    fftw_execute_r2r(dct2_plan[std::make_pair(x.shape(0), x.shape(1))], x.data(), x.data());


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