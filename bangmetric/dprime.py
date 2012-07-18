"""D' (d-prime) Sensitivity Index"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Ha Hong <hahong84@gmail.com>
#
# License: BSD

__all__ = ['dprime', 'dprime_from_confusion_ova']

import numpy as np
from scipy.stats import norm

DEFAULT_FUDGE_FACTOR = 0.5
DEFAULT_FUDGE_MODE = 'correction'
ATOL = 1e-6


def dprime(y_pred, y_true, **kwargs):
    """Computes the d-prime sensitivity index of the predictions.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values, interpreted as strictly positive or not
        (i.e. converted to binary).
        Could be in {-1, +1} or {0, 1} or {False, True}.

    y_pred: array, shape = [n_samples]
        Predicted values (real).

    kwargs: named arguments, optional
        Passed to ``dprime_from_samp()``.

    Returns
    -------
    dp: float or None
        d-prime, None if d-prime is undefined and raw d-prime value (``safedp=False``)
        is not requested (default).

    References
    ----------
    http://en.wikipedia.org/wiki/D'
    """

    # -- basic checks and conversion
    assert len(y_true) == len(y_pred)
    assert np.isfinite(y_true).all()
    assert np.isfinite(y_pred).all()

    y_true = np.array(y_true)
    assert y_true.ndim == 1

    y_pred = np.array(y_pred)
    assert y_pred.ndim == 1

    # -- actual computation
    i_pos = y_true > 0
    i_neg = ~i_pos

    pos = y_pred[i_pos]
    neg = y_pred[i_neg]

    dp = dprime_from_samp(pos, neg, bypass_nchk=True, **kwargs)
    return dp


def dprime_from_samp(pos, neg, maxv=None, minv=None, safedp=True, bypass_nchk=False):
    """Computes the d-prime sensitivity index from positive and negative samples.

    Parameters
    ----------
    pos: array-like
        Positive sample values (e.g., raw projection values of the positive classifier).

    neg: array-like
        Negative sample values.

    maxv: float, optional
        Maximum possible d-prime value. If None (default), there's no limit on
        the maximum value.

    minv: float, optional
        Minimum possible d-prime value. If None (default), there's no limit.

    safedp: bool, optional
        If True (default), this function will return None if the resulting d-prime 
        value becomes non-finite.

    bypass_nchk: bool, optional
        If False (default), do not bypass the test to ensure that enough positive 
        and negatives samples are there for the variance estimation.

    Returns
    -------
    dp: float or None
        d-prime, None if d-prime is undefined and raw d-prime value (``safedp=False``)
        is not requested (default).

    References
    ----------
    http://en.wikipedia.org/wiki/D'
    """

    pos = np.array(pos)
    neg = np.array(neg)

    if not bypass_nchk:
        assert pos.size > 1, 'Not enough positive samples to estimate the variance'
        assert neg.size > 1, 'Not enough negative samples to estimate the variance'

    pos_mean = pos.mean()
    neg_mean = neg.mean()
    pos_var = pos.var(ddof=1)
    neg_var = neg.var(ddof=1)

    num = pos_mean - neg_mean
    div = np.sqrt((pos_var + neg_var) / 2.)

    # from Dan's suggestion about clipping d' values...
    if maxv is None:
        maxv = np.inf
    if minv is None:
        minv = -np.inf

    dp = np.clip(num / div, minv, maxv)

    if safedp and not np.isfinite(dp):
        dp = None

    return dp


def dprime_from_confusion_ova(M, fudge_mode=DEFAULT_FUDGE_MODE, \
        fudge_fac=DEFAULT_FUDGE_FACTOR, atol=ATOL):
    """Computes the one-vs-all d-prime sensitivity index of the confusion matrix.

    Parameters
    ----------
    M: array, shape = [n_classes (true), n_classes (pred)] 
        Confusion matrix, where the element M_{rc} means the number of
        times when the classifier guesses that a test sample in the r-th class
        belongs to the c-th class.

    fudge_fac: float, optional
        A small factor to avoid non-finite numbers when TPR or FPR becomes 0 or 1.

    fudge_mode: str, optional
        Determins how to apply the fudge factor
            'always': always apply the fudge factor 
            'correction': apply only when needed

    atol: float, optional
        Tolerance to simplify the dp from a  2-way (i.e., 2x2) confusion matrix.

    Returns
    -------
    dp: array, shape = [n_classes]
        Array of d-primes, each element corresponding to each class

    References
    ----------
    http://en.wikipedia.org/wiki/D'
    http://en.wikipedia.org/wiki/Confusion_matrix

    XXX: no normalization for unbalanced data
    """

    M = np.array(M)
    assert M.ndim == 2
    assert M.shape[0] == M.shape[1]
    
    P = np.sum(M, axis=1)   # number of positives, for each class
    N = np.sum(P) - P

    TP = np.diag(M)
    FP = np.sum(M, axis=0) - TP

    if fudge_mode == 'always':    # always apply fudge factor
        TPR = (TP.astype('float') + fudge_fac) / (P + 2.*fudge_fac)
        FPR = (FP.astype('float') + fudge_fac) / (N + 2.*fudge_fac)

    elif fudge_mode == 'correction':   # apply fudge factor only when needed
        TP = TP.astype('float')
        FP = FP.astype('float')

        TP[TP == P] = P[TP == P] - fudge_fac    # 100% correct
        TP[TP == 0] = fudge_fac                 # 0% correct
        FP[FP == N] = N[FP == N] - fudge_fac    # always FAR
        FP[FP == 0] = fudge_fac                 # no false alarm

        TPR = TP / P
        FPR = FP / N

    else:
        assert False, 'Not implemented'

    dp = norm.ppf(TPR) - norm.ppf(FPR)
    # if there's only two dp's then, it's must be "A" vs. "~A" task.  If so, just give one value
    if len(dp) == 2 and np.abs(dp[0] - dp[1]) < atol:
        dp = np.array([dp[0]])

    return dp

