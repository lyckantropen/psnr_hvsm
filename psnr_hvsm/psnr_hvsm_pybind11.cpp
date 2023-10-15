#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include "dct.h"
#include "psnr_hvsm.h"
#include "hvsm.h"

PYBIND11_MODULE(_psnr_hvsm, m)
{
  namespace py = pybind11;
  xt::import_numpy();

  m.doc() = "Accelerated implementation of the PSNR-HVS and PSNR-HVS-M image metrics.";
  m.def(
      "dct", [](const xt::pytensor<double, 1>& x, bool norm) -> xt::pytensor<double, 1>
      { return xt::pytensor<double, 1>{psnr_hvsm::dct(x, norm)}; },
      py::arg("x"), py::arg("norm") = true);
  m.def(
      "dct2", [](const xt::pytensor<double, 2>& x, bool norm) -> xt::pytensor<double, 2>
      { return xt::pytensor<double, 2>{psnr_hvsm::dct2(x, norm)}; },
      py::arg("x"), py::arg("norm") = true);
  m.def(
      "hvs_hvsm_mse", [](const xt::pytensor<double, 2>& x, const xt::pytensor<double, 2>& y) -> std::pair<double, double>
      { return psnr_hvsm::hvs_hvsm_mse(x, y); },
      py::arg("image_a"), py::arg("image_b"));
  m.def(
      "hvs_hvsm_mse_tiles", [](const xt::pytensor<double, 2>& x, const xt::pytensor<double, 2>& y) -> std::pair<xt::pytensor<double, 2>, xt::pytensor<double, 2>>
      { return psnr_hvsm::hvs_hvsm_mse_tiles(x, y); },
      py::arg("image_a"), py::arg("image_b"));
  m.def(
      "masking", [](const xt::pytensor<double, 2>& x, const xt::pytensor<double, 2>& y) -> double
      { return psnr_hvsm::masking<xt::xtensor<double, 2>, xt::xtensor<double, 2>>(x, y); },
      py::arg("tile"), py::arg("tile_dct"));
  m.def(
      "psnr_hvs_hvsm", [](const xt::pytensor<double, 2>& x, const xt::pytensor<double, 2>& y) -> std::pair<double, double>
      { return psnr_hvsm::psnr_hvs_hvsm(x, y); },
      py::arg("image_a"), py::arg("image_b"));
}