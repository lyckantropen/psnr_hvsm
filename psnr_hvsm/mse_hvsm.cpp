#include "mse_hvsm.h"

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include "mse_hvs.h"
#include "dct.h"

using xt::placeholders::_;

double masking(const xt::xtensor<double, 2> &tile, const xt::xtensor<double, 2> &tile_dct)
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

double hvsm_mse_tile(const xt::xtensor<double, 2> &tile_a, const xt::xtensor<double, 2> &tile_b)
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

  return weighted_mse / double(DCT_H * DCT_W);
}

xt::xtensor<double, 2> hvsm_mse(const xt::xtensor<double, 2> &image_a, const xt::xtensor<double, 2> &image_b)
{
  const size_t blocks_y = image_a.shape(0) / DCT_H;
  const size_t blocks_x = image_a.shape(1) / DCT_W;
  xt::xtensor<double, 2> mse_per_tile = xt::zeros<double>({blocks_y, blocks_x});

  for (size_t y = 0; y < image_a.shape(0); y += DCT_H)
  {
    for (size_t x = 0; x < image_a.shape(1); x += DCT_W)
    {
      mse_per_tile(y / DCT_H, x / DCT_W) = hvsm_mse_tile(xt::view(image_a, xt::range(y, y + DCT_H), xt::range(x, x + DCT_W)), xt::view(image_b, xt::range(y, y + DCT_H), xt::range(x, x + DCT_W)));
    }
  }
  return mse_per_tile;
}
