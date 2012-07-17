"""D' (d-prime) Sensitivity Index"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Ha Hong <hahong84@gmail.com>
#
# License: BSD

__all__ = ['dprime', 'dprime_ova_from_confusion']

import numpy as np
from scipy.stats import norm

DEFAULT_FUDGE_FACTOR = 0.5
DEFAULT_FUDGE_MODE = 'correction'
ATOL = 1e-7

def dprime(y_pred, y_true):
    """Computes the d-prime sensitivity index of the predictions.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values, interpreted as strictly positive or not
        (i.e. converted to binary).
        Could be in {-1, +1} or {0, 1} or {False, True}.

    y_pred: array, shape = [n_samples]
        Predicted values, interpreted as strictly positive or not
        (i.e. converted to binary).

    Returns
    -------
    dp: float or None
        d-prime, None if d-prime is undefined

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
    pos = y_true > 0
    neg = ~pos
    pos_mean = y_pred[pos].mean()
    neg_mean = y_pred[neg].mean()
    pos_var = y_pred[pos].var(ddof=1)
    neg_var = y_pred[neg].var(ddof=1)

    num = pos_mean - neg_mean
    div = np.sqrt((pos_var + neg_var) / 2.)
    if div == 0:
        dp = None
    else:
        dp = num / div

    return dp


def dprime_ova_from_confusion(M, fudge_mode=DEFAULT_FUDGE_MODE, \
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

