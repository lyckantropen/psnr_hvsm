#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include "dct.h"
#include "mse_hvs.h"
#include "mse_hvsm.h"
#include "psnr_hvsm.h"

namespace py = pybind11;

PYBIND11_MODULE(_psnr_hvsm, m)
{
  xt::import_numpy();

  m.doc() = "Accelerated implementation of the PSNR-HVS and PSNR-HVS-M image metrics.";
  m.def(
      "dct", [](xt::pytensor<double, 1> x, bool norm) -> xt::pytensor<double, 1> {
        return xt::pytensor<double, 1>{dct(x, norm)};
      },
      py::arg("x"), py::arg("norm") = true);
  m.def(
      "dct2", [](xt::pytensor<double, 2> x, bool norm) -> xt::pytensor<double, 2> {
        return xt::pytensor<double, 2>{dct2(x, norm)};
      },
      py::arg("x"), py::arg("norm") = true);
  m.def(
      "hvs_mse_tile", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return hvs_mse_tile(x, y);
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "hvs_mse", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> xt::pytensor<double, 2> {
        return xt::pytensor<double, 2>{hvs_mse(x, y)};
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "masking", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return masking(x, y);
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "hvsm_mse_tile", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return hvsm_mse_tile(x, y);
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "hvsm_mse", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> xt::pytensor<double, 2> {
        return xt::pytensor<double, 2>{hvsm_mse(x, y)};
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "psnr_hvs", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return psnr_hvs(x, y);
      },
      py::arg("x"), py::arg("y"));
  m.def(
      "psnr_hvsm", [](xt::pytensor<double, 2> x, xt::pytensor<double, 2> y) -> double {
        return psnr_hvsm(x, y);
      },
      py::arg("x"), py::arg("y"));
}