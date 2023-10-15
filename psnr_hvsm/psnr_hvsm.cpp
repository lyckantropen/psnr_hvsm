#include "psnr_hvsm.h"
#include "hvsm.h"

#include <cmath>

namespace psnr_hvsm
{
  namespace detail
  {
    double psnr(double mse)
    {
      if (mse == 0.0)
      {
        return 100.0;
      }
      else
      {
        return std::min(10.0 * std::log10(1.0 / mse), 100.0);
      }
    }
  }

  std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>> hvs_hvsm_mse_tiles(const xt::xtensor<double, 2> &image_a, const xt::xtensor<double, 2> &image_b)
  {
    const size_t blocks_y = image_a.shape(0) / DCT_H;
    const size_t blocks_x = image_a.shape(1) / DCT_W;
    xt::xtensor<double, 2> hvs_mse_per_tile = xt::zeros<double>({blocks_y, blocks_x});
    xt::xtensor<double, 2> hvsm_mse_per_tile = xt::zeros<double>({blocks_y, blocks_x});

    #ifdef PSNR_HVSM_OPENMP
    #pragma omp parallel for
    #endif
    for (int y = 0; y < image_a.shape(0); y += DCT_H)
    {
      for (int x = 0; x < image_a.shape(1); x += DCT_W)
      {
        std::tie(
            hvs_mse_per_tile(y / DCT_H, x / DCT_W),
            hvsm_mse_per_tile(y / DCT_H, x / DCT_W)) =
            hvs_hvsm_mse_tile(
                xt::view(image_a, xt::range(y, y + DCT_H), xt::range(x, x + DCT_W)),
                xt::view(image_b, xt::range(y, y + DCT_H), xt::range(x, x + DCT_W)));
      }
    }
    return std::make_pair(hvs_mse_per_tile, hvsm_mse_per_tile);
  }

  std::pair<double,double> hvs_hvsm_mse(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
  {
    auto [hvs, hvsm] = hvs_hvsm_mse_tiles(a, b);
    return std::make_pair(xt::mean(std::move(hvs))(), xt::mean(std::move(hvsm))());
  }

  std::pair<double, double> psnr_hvs_hvsm(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
  {
    auto [hvs, hvsm] = hvs_hvsm_mse_tiles(a, b);
    return std::make_pair(detail::psnr(xt::mean(std::move(hvs))()), detail::psnr(xt::mean(std::move(hvsm))()));
  }
}