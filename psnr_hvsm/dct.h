#pragma once
#include <xtensor/xtensor.hpp>

namespace psnr_hvsm
{
  xt::xtensor<double, 1> dct(xt::xtensor<double, 1> x, bool norm = true);
  xt::xtensor<double, 2> dct2(xt::xtensor<double, 2> x, bool norm = true);
}
