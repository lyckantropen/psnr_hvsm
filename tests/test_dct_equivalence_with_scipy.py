import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.fft import dct as sp_dct
from scipy.fft import dctn as sp_dctn

from psnr_hvsm import dct, dct2


def test_dct_produces_same_result_on_1d_ramp():
    x = np.arange(16, dtype=float)

    assert_array_almost_equal(dct(x), sp_dct(x, type=2))


def test_dct2_produces_same_result_on_2d_ramp():
    x = np.arange(64, dtype=float).reshape(8, 8)

    assert_array_almost_equal(dct2(x), sp_dctn(x, type=2))


def test_dct_produces_same_result_on_1d_assorted_values():
    x = np.array([1.16517274, -1.13907045, -1.83350938, -0.97573503, -1.08947107,
                  0.57221949, -1.78049922, -0.06325282, -1.46966454, -0.22084793,
                  1.89964033, -0.21133256, 1.45662116, -1.9070239, 1.03049493,
                  1.81084522])

    assert_array_almost_equal(dct(x), sp_dct(x, type=2))


def test_dct2_produces_same_result_on_2d_assorted_values():
    x = np.array([[-0.27268473, 1.89427747, -0.28792127, -2.94728155, 3.1352648,
                   -1.29961984, 0.31517309, -1.79968345],
                  [3.01363283, 0.54625221, 3.47701026, -3.8350829, -5.68073552,
                   -2.74884637, -1.5547372, 5.37038163],
                  [5.69264403, -2.09493016, 0.19327069, 3.95588185, 2.10646164,
                   4.89162528, -2.37808545, -3.93939287],
                  [4.7639939, 2.2113092, 4.63637059, 1.11217797, 5.31258127,
                   -0.7039605, -4.48682271, -2.0053419],
                  [4.86195827, 2.02764099, -0.01199882, -4.1180301, 0.82714978,
                   3.51884827, 5.0710077, 2.83230969],
                  [5.60743798, -4.65218129, -1.12208591, -4.80151406, -1.26609585,
                   -3.7500127, -0.042037, -0.45495029],
                  [-5.44219476, 2.90493246, 5.83133002, -1.5749625, -1.89951152,
                   2.48281233, -1.27051191, -0.86414043],
                  [-2.44960754, 4.17607207, -1.38446204, 1.34689231, -0.48578666,
                   -3.4962453, 1.8764909, -5.84631864]])

    assert_array_almost_equal(dct2(x), sp_dctn(x, type=2))
