"""Other utility functions"""

# Authors: Ha Hong <hahong84@gmail.com>
#
# License: BSD

__all__ = ['confusion_stats']

import numpy as np

DEFAULT_FUDGE_FACTOR = 0.5
DEFAULT_FUDGE_MODE = 'correction'


def confusion_stats(M, collation=None, \
        fudge_mode=DEFAULT_FUDGE_MODE, fudge_factor=DEFAULT_FUDGE_FACTOR):
    """Computes classification statistics of sub-confusion matrices inside 
    the given original confusion matrix M.  If no ``collation`` is given,
    statistics for each one vs. rest sub-confusion matrix will be computed.

    Parameters
    ----------
    M: array-like, shape = [n_classes (true), n_classes (pred)] 
        Confusion matrix, where the element M_{rc} means the number of
        times when the classifier guesses that a test sample in the r-th class
        belongs to the c-th class.

    collation: None or array-like with shape = [n_groupings, n_classes], optional
        Defines how to group entries in `M` to compute TPR and FPR.  
        Entries shoule be {+1, 0, -1}.  A row defines one instance of grouping,
        where +1, -1, and 0 designate the corresponding class as a
        positive, negative, and ignored class, respectively.  For example, 
        the following `collation` defines a 3-way one vs. rest grouping 
        (given that `M` is a 3x3 matrix):
            [[+1, -1, -1],
             [-1, +1, -1],
             [-1, -1, +1]]
        If `None` (default), one vs. rest grouping is assumed.

    fudge_factor: float, optional
        A small factor to avoid non-finite numbers when TPR or FPR becomes 0 or 1.
        Default is 0.5.

    fudge_mode: str, optional
        Determins how to apply the fudge factor.  Can be one of:
            'correction': apply only when needed (default)
            'always': always apply the fudge factor
            'none': no fudging --- equivalent to ``fudge_factor=0``


    Returns
    -------
    P: array, shape = [n_groupings]
        Array of the number of positives, where each element corresponds to each 
        grouping defined by `collation`.
    N: array, shape = [n_groupings]
        Same as P, except that this is an array of the number of negatives.
    TP: array, shape = [n_groupings]
        Same as P, except that this is an array of the number of true positives.
    TN: array, shape = [n_groupings]
        Same as P, except that this is an array of the number of true negatives.
    FP: array, shape = [n_groupings]
        Same as P, except that this is an array of the number of false positives.
    FN: array, shape = [n_groupings]
        Same as P, except that this is an array of the number of false negatives.


    References
    ----------
    http://en.wikipedia.org/wiki/Confusion_matrix
    http://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """

    # M: confusion matrix, row means true classes, col means predicted classes
    M = np.array(M)
    assert M.ndim == 2
    assert M.shape[0] == M.shape[1]
    n_classes = M.shape[0]

    if collation is None:    
        # make it one vs. rest
        collation = -np.ones((n_classes, n_classes), dtype='int8')
        collation += 2 * np.eye(n_classes, dtype='int8')
    else:
        collation = np.array(collation, dtype='int8')
        assert collation.ndim == 2
        assert collation.shape[1] == n_classes
    
    # P0: number of positives, for each class
    # P: number of positives, for each grouping
    # N: number of negatives, for each grouping
    # TP: number of true positives, for each grouping
    # FP: number of false positives, for each grouping
    P0 = np.sum(M, axis=1)   
    P = np.array([np.sum(P0[coll == +1]) for coll in collation], dtype='float64')
    N = np.array([np.sum(P0[coll == -1]) for coll in collation], dtype='float64')
    TP = np.array([np.sum(M[coll == +1][:, coll == +1]) for coll in collation], dtype='float64')
    TN = np.array([np.sum(M[coll == -1][:, coll == -1]) for coll in collation], dtype='float64')
    FP = np.array([np.sum(M[coll == -1][:, coll == +1]) for coll in collation], dtype='float64')
    FN = np.array([np.sum(M[coll == +1][:, coll == -1]) for coll in collation], dtype='float64')

    # -- application of fudge factor
    if fudge_mode == 'none':           # no fudging
        pass

    elif fudge_mode == 'always':       # always apply fudge factor
        TP += fudge_factor
        FP += fudge_factor
        TN += fudge_factor
        FN += fudge_factor
        P += 2.*fudge_factor
        N += 2.*fudge_factor

    elif fudge_mode == 'correction':   # apply fudge factor only when needed
        TP[TP == P] = P[TP == P] - fudge_factor    # 100% correct
        TP[TP == 0] = fudge_factor                 # 0% correct
        FP[FP == N] = N[FP == N] - fudge_factor    # always FAR
        FP[FP == 0] = fudge_factor                 # no false alarm

        TN[TN == N] = N[TN == N] - fudge_factor    
        TN[TN == 0] = fudge_factor                 
        FN[FN == P] = P[FN == P] - fudge_factor    
        FN[FN == 0] = fudge_factor                 

    else:
        raise ValueError('Invalid fudge_mode')

    # -- done
    return P, N, TP, TN, FP, FN

