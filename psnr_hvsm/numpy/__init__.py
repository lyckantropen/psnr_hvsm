PRECISIONS = ('8-bit', 'float')


def validate_precision(precision):
    if precision not in PRECISIONS:
        raise ValueError("precision must be one of %r, got %r" % (PRECISIONS, precision))


from .bt601 import *  # noqa
from .psnr import *  # noqa
from .psnr_hma import *  # noqa
from .psnr_hvsm import *  # noqa
