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

    std::tuple<xt::xtensor<double, 2>, xt::xtensor<double, 2>> hvs_hvsm_mse(const xt::xtensor<double, 2> &image_a, const xt::xtensor<double, 2> &image_b)
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
      return std::make_tuple(hvs_mse_per_tile, hvsm_mse_per_tile);
    }
  }

  double hvs_mse_tile(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
  {
    return std::get<0>(hvs_hvsm_mse_tile(a, b));
  }

  double hvsm_mse_tile(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
  {
    return std::get<1>(hvs_hvsm_mse_tile(a, b));
  }

  xt::xtensor<double, 2> hvs_mse(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
  {
    return std::get<0>(detail::hvs_hvsm_mse(a, b));
  }

  xt::xtensor<double, 2> hvsm_mse(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
  {
    return std::get<1>(detail::hvs_hvsm_mse(a, b));
  }

  std::tuple<double, double> psnr_hvs_hvsm(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
  {
    auto mse = detail::hvs_hvsm_mse(a, b);
    return std::make_tuple(detail::psnr(xt::mean(std::get<0>(mse))()), detail::psnr(xt::mean(std::get<1>(mse))()));
  }

  double psnr_hvs(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
  {
    return std::get<0>(psnr_hvs_hvsm(a, b));
  }

  double psnr_hvsm(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b)
  {
    return std::get<1>(psnr_hvs_hvsm(a, b));
  }
}