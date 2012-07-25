"""Metrics designed to compute the similarity to human data"""

# Authors: Ha Hong <hahong84@gmail.com>
#
# License: BSD

__all__ = ['central_ratio', 'consistency']

import numpy as np
from .correlation import spearman

DTYPE = np.float64


def central_ratio(num, dnm, centerfn=np.median, finite=True):
    """Computes the central tendency (median, by default) of the ratios
    between `num` and `dnm`.  By default, this function gives the
    "Turing ratio" used in the paper by Majaj, Hong, Solomon, and DiCarlo.

    Parameters
    ----------
    num: array-like
        Numerators of ratios

    dnm: array-like, shape = `num.shape()`
        Denominators of ratios.  `num` and `dnm` must have the same shape.

    centerfn: function, optional (default=np.median)
        Function to compute the central tendency.

    finite: boolean, optional (default=True)
        If True, only finite numbers in `num` and `dnm` will be used for
        the computation of the central tendency.
    """

    num = np.array(num, dtype=DTYPE)
    dnm = np.array(dnm, dtype=DTYPE)
    assert num.shape == dnm.shape

    num = num.ravel()
    dnm = dnm.ravel()

    if finite:
        fi = np.isfinite(dnm) & np.isfinite(num)
        num = num[fi]
        dnm = dnm[fi]

    return centerfn(num / dnm)


def consistency(A, B, consistencyfn=spearman, finite=True):
    """Computes the consistency (Spearman rank correlation coefficient,
    by default) between two sets of data points (e.g., d' scores) `A`
    and `B`.  By default, this function gives the "consistency"
    used in the paper by Majaj, Hong, Solomon, and DiCarlo.

    Parameters
    ----------
    A: array-like
        A set of data points

    B: array-like, shape = `A.shape()`
        Another set of data points to compare with `A`.
        `A` and `B` must have the same shape.

    consistencyfn: function, optional (default=bangmetric.spearman)
        Function to compute the "consistency."

    finite: boolean, optional (default=True)
        If True, only finite numbers in `A` and `B` will be used for
        the computation of the consistency.
    """

    A = np.array(A, dtype=DTYPE)
    B = np.array(B, dtype=DTYPE)
    assert A.shape == B.shape

    A = A.ravel()
    B = B.ravel()

    if finite:
        fi = np.isfinite(B) & np.isfinite(A)
        A = A[fi]
        B = B[fi]

    return consistencyfn(A, B)
