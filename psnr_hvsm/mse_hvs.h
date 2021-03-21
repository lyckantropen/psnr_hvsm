#pragma once
#include <xtensor/xtensor.hpp>

constexpr double CSF_COEFF[64] =
    {1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887,
     2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911,
     1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555,
     1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082,
     1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222,
     1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729,
     0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803,
     0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950};
constexpr size_t DCT_H = 8;
constexpr size_t DCT_W = 8;

double hvs_mse_tile(const xt::xtensor<double, 2> &, const xt::xtensor<double, 2> &);
xt::xtensor<double, 2> hvs_mse(const xt::xtensor<double, 2> &, const xt::xtensor<double, 2> &);