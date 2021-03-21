#pragma once
#include <xtensor/xtensor.hpp>

double psnr_hvs(const xt::xtensor<double,2>& a, const xt::xtensor<double,2>&b);
double psnr_hvsm(const xt::xtensor<double,2>& a, const xt::xtensor<double,2>&b);
