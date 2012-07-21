"""D' (d-prime) Sensitivity Index"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Ha Hong <hahong84@gmail.com>
#
# License: BSD

__all__ = ['dprime', 'dprime_from_confusion_matrix']

import numpy as np
from scipy.stats import norm
from .utils import confusion_stats


def dprime(y_pred=None, y_true=None, pos=None, neg=None, max_value=np.inf, min_value=-np.inf):
    """Computes the d-prime sensitivity index.
    One must provide either y_pred and y_true or pos and neg.
    This function computes the d-prime of predictions given by 
    y_pred and y_true by default. If pos and neg are provided,
    both y_pred and y_true are ignored, and this function 
    computes the d-prime from positive and negative samples
    given by pos and neg.

    Parameters
    ----------
    y_true: array, shape = [n_samples], optional
        True values, interpreted as strictly positive or not
        (i.e. converted to binary).
        Could be in {-1, +1} or {0, 1} or {False, True}.

    y_pred: array, shape = [n_samples], optional
        Predicted values (real).

    pos: array-like, optional
        Positive sample values (e.g., raw projection values of 
        the positive classifier).

    neg: array-like, optional
        Negative sample values. If both pos and neg are
        provided, y_true and y_pred are ignored.

    max_value: float, optional
        Maximum possible d-prime value. Default is ``np.inf``.

    min_value: float, optional
        Minimum possible d-prime value. Default is ``-np.inf``.

    Returns
    -------
    dp: float
        d-prime

    References
    ----------
    http://en.wikipedia.org/wiki/D'
    """

    # -- basic checks and conversion
    if pos is not None and neg is not None:
        pos = np.array(pos)
        neg = np.array(neg)

    else:
        assert len(y_true) == len(y_pred)
        assert np.isfinite(y_true).all()

        y_true = np.array(y_true)
        assert y_true.ndim == 1

        y_pred = np.array(y_pred)
        assert y_pred.ndim == 1

        # -- actual computation
        i_pos = y_true > 0
        i_neg = ~i_pos

        pos = y_pred[i_pos]
        neg = y_pred[i_neg]

    assert np.isfinite(pos).all()
    assert np.isfinite(neg).all()

    if pos.size <= 1:
        raise ValueError('Not enough positive samples to estimate the variance')
    if neg.size <= 1:
        raise ValueError('Not enough negative samples to estimate the variance')

    # -- compute d'
    pos_mean = pos.mean()
    neg_mean = neg.mean()
    pos_var = pos.var(ddof=1)
    neg_var = neg.var(ddof=1)

    num = pos_mean - neg_mean
    div = np.sqrt((pos_var + neg_var) / 2.)

    # from Dan's suggestion about clipping d' values...
    dp = np.clip(num / div, min_value, max_value)

    return dp


def dprime_from_confusion_matrix(M, max_value=np.inf, min_value=-np.inf, **kwargs):
    """Computes the d-prime sensitivity indices of the given confusion matrix.
    This function is designed mostly for when there is no access to internal 
    representations and/or decision making mechanisms (like human data).  
    If no ``collation`` is defined in ``kwargs`` this function computes 
    one vs. rest d-prime for each class.

    Parameters
    ----------
    M: array-like, shape = [n_classes (true), n_classes (pred)] 
        Confusion matrix, where the element M_{rc} means the number of
        times when the classifier/subject guesses that a test sample in 
        the r-th class belongs to the c-th class.

    max_value: float, optional
        Maximum possible d-prime value. Default is ``np.inf``.

    min_value: float, optional
        Minimum possible d-prime value. Default is ``-np.inf``.

    kwargs: named arguments, optional
        Passed to ``confusion_stats()``.  By passing ``collation``, ``fudge_mode``,
        ``fudge_factor``, etc. one can change the behavior of d-prime computation 
        (see ``confusion_stats()`` for details). 

    Returns
    -------
    dp: array, shape = [n_groupings]
        Array of d-primes, where each element corresponds to each grouping
        defined by `collation` (see ``confusion_stats()`` for details).

    References
    ----------
    http://en.wikipedia.org/wiki/D'
    http://en.wikipedia.org/wiki/Confusion_matrix
    """

    # M: confusion matrix, row means true classes, col means predicted classes
    P, N, TP, _, FP, _ = confusion_matrix_stats(M, **kwargs)

    TPR = TP / P
    FPR = FP / N
    dp = np.clip(norm.ppf(TPR) - norm.ppf(FPR), min_value, max_value)

    return dp

