#pragma once
#include <xtensor/xtensor.hpp>
#include <tuple>

namespace psnr_hvsm
{
  std::pair<double,double> hvs_hvsm_mse(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b);
  std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>> hvs_hvsm_mse_tiles(const xt::xtensor<double, 2> &image_a, const xt::xtensor<double, 2> &image_b);
  std::pair<double, double> psnr_hvs_hvsm(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b);
}