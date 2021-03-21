#pragma once
#include <xtensor/xtensor.hpp>

xt::xtensor<double, 1> dct(const xt::xtensor<double, 1> &x);
xt::xtensor<double, 2> dct2(const xt::xtensor<double, 2> &x);
