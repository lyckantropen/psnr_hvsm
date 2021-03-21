#include "psnr_hvsm.h"

#include <cmath>
#include "mse_hvs.h"
#include "mse_hvsm.h"

template <typename _GetMse>
double psnr(_GetMse&& mse_f, const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
{
  auto mse = xt::mean(mse_f(a, b))();
  if (mse == 0.0)
  {
    return 100.0;
  }
  else
  {
    return std::min(10.0 * std::log10(1.0 / mse), 100.0);
  }
}

double psnr_hvs(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
{
  return psnr(hvs_mse, a, b);
}
double psnr_hvsm(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
{
  return psnr(hvsm_mse, a, b);
}
