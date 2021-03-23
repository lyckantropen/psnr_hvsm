#pragma once
#include <xtensor/xtensor.hpp>
#include <tuple>

namespace psnr_hvsm
{
  double hvs_mse_tile(const xt::xtensor<double, 2> &, const xt::xtensor<double, 2> &);
  double hvsm_mse_tile(const xt::xtensor<double, 2> &, const xt::xtensor<double, 2> &);
  xt::xtensor<double, 2> hvs_mse(const xt::xtensor<double, 2> &, const xt::xtensor<double, 2> &);
  xt::xtensor<double, 2> hvsm_mse(const xt::xtensor<double, 2> &, const xt::xtensor<double, 2> &);
  std::tuple<double, double> psnr_hvs_hvsm(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b);
  double psnr_hvs(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b);
  double psnr_hvsm(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b);
}