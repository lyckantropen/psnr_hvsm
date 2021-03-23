#pragma once
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>

#include "dct.h"

namespace psnr_hvsm
{

  constexpr double MASK_COEFF[64] =
      {0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874,
       0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058,
       0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888,
       0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015,
       0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866,
       0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815,
       0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803,
       0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203};
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

  using xt::placeholders::_;

  template <class _XExpressionA, class _XExpressionB>
  double masking(_XExpressionA &&tile, _XExpressionB &&tile_dct)
  {
    const size_t qh = DCT_H / 2;
    const size_t qw = DCT_W / 2;
    const auto mask_coeff = xt::adapt(MASK_COEFF, {DCT_H * DCT_W});

    const auto acs = xt::view(xt::flatten(tile_dct), xt::range(1, _));
    const double mask = xt::sum(xt::pow(acs, 2) * xt::view(mask_coeff, xt::range(1, _)))();

    auto vari = [](auto &&a) {
      return (xt::variance(a, 1) * a.size())();
    };

    double var = vari(tile);
    if (var != 0.0)
    {
      var = (vari(xt::view(tile, xt::range(0, qh), xt::range(0, qw))) +
             vari(xt::view(tile, xt::range(0, qh), xt::range(qw, _))) +
             vari(xt::view(tile, xt::range(qh, _), xt::range(0, qw))) +
             vari(xt::view(tile, xt::range(qh, _), xt::range(qw, _)))) /
            var;
    }
    return std::sqrt(mask * var / double(qh * qw) / double(DCT_H * DCT_W));
  }

  template <class _XExpressionA, class _XExpressionB>
  std::tuple<double, double> hvs_hvsm_mse_tile(_XExpressionA &&tile_a, _XExpressionB &&tile_b)
  {
    const auto mask_coeff = xt::adapt(MASK_COEFF, {DCT_H, DCT_W});
    const auto coeff = xt::adapt(CSF_COEFF, {DCT_H, DCT_W});
    const auto dct_a = dct2(tile_a);
    const auto dct_b = dct2(tile_b);

    const auto dif = xt::abs(dct_a - dct_b);
    double mask_a = masking(tile_a, dct_a);
    double mask_b = masking(tile_b, dct_b);

    if (mask_b > mask_a)
    {
      mask_a = mask_b;
    }

    double weighted_mse = std::pow(dif(0, 0) * coeff(0, 0), 2);
    auto mask = xt::pow(mask_coeff, -1.0) * mask_a;
    auto masked_dct = xt::eval(xt::where(dif >= mask, dif - mask, 0.0) * coeff);
    auto masked_dct_sq = xt::pow(xt::view(xt::flatten(masked_dct), xt::range(1, _)), 2);
    weighted_mse += xt::sum(masked_dct_sq)();

    return std::make_tuple((xt::sum(xt::pow(dif * coeff, 2)) / double(DCT_H * DCT_W))(), weighted_mse / double(DCT_H * DCT_W));
  }

}