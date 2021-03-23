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
      "dct", [](xt::pytensor<double, 1> x, bool norm) -> xt::pytensor<double, 1> {
        return xt::pytensor<double, 1>{psnr_hvsm::dct(x, norm)};
      },
      py::arg("x"), py::arg("norm") = true);
  m.def(
      "dct2", [](xt::pytensor<double, 2> x, bool norm) -> xt::pytensor<double, 2> {
        return xt::pytensor<double, 2>{psnr_hvsm::dct2(x, norm)};
      },
      py::arg("x"), py::arg("norm") = true);
  m.def(
      "hvs_mse_tile", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return psnr_hvsm::hvs_mse_tile(x, y);
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "hvs_mse", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> xt::pytensor<double, 2> {
        return xt::pytensor<double, 2>{psnr_hvsm::hvs_mse(x, y)};
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "masking", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return psnr_hvsm::masking<xt::xtensor<double, 2>, xt::xtensor<double, 2>>(x, y);
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "hvsm_mse_tile", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return psnr_hvsm::hvsm_mse_tile(x, y);
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "hvsm_mse", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> xt::pytensor<double, 2> {
        return xt::pytensor<double, 2>{psnr_hvsm::hvsm_mse(x, y)};
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "psnr_hvs", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return psnr_hvsm::psnr_hvs(x, y);
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "psnr_hvsm", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return psnr_hvsm::psnr_hvsm(x, y);
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "psnr_hvs_hvsm", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> std::tuple<double, double> {
        return psnr_hvsm::psnr_hvs_hvsm(x, y);
      },
      py::arg("x"), py::arg("y"));
}