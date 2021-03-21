#include "mse_hvs.h"
#include "dct.h"

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

double hvs_mse_tile(const xt::xtensor<double, 2> &tile_a, const xt::xtensor<double, 2> &tile_b)
{
  const auto coeff = xt::adapt(CSF_COEFF, {DCT_H, DCT_W});
  const auto dct_a = dct2(tile_a);
  const auto dct_b = dct2(tile_b);

  const auto dif = xt::abs(dct_a - dct_b);
  return (xt::sum(xt::pow(dif * coeff, 2)) / double(DCT_H*DCT_W))();
}

xt::xtensor<double, 2> hvs_mse(const xt::xtensor<double, 2> &image_a, const xt::xtensor<double, 2> &image_b)
{
  const size_t blocks_y = image_a.shape(0) / DCT_H;
  const size_t blocks_x = image_a.shape(1) / DCT_W;
  xt::xtensor<double, 2> mse_per_tile = xt::zeros<double>({blocks_y, blocks_x});

  for (size_t y = 0; y < image_a.shape(0); y += DCT_H)
  {
    for (size_t x = 0; x < image_a.shape(1); x += DCT_W)
    {
      mse_per_tile(y / DCT_H, x / DCT_W) = hvs_mse_tile(xt::view(image_a, xt::range(y, y + DCT_H), xt::range(x, x + DCT_W)), xt::view(image_b, xt::range(y, y + DCT_H), xt::range(x, x + DCT_W)));
    }
  }
  return mse_per_tile;
}