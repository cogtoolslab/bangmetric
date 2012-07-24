"""Accuracy"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['accuracy']

import numpy as np
from .utils import confusion_matrix_stats

DEFAULT_ACCURACY_MODE = 'binary'


def accuracy(A, B=None, mode=DEFAULT_ACCURACY_MODE, \
        balanced=False, collation=None):
    """Computes the accuracy of the predictions (also known as the
    zero-one score).  Depending on the choice of `mode`, this
    function can take one of the following data format:

    * Binary classification outputs (`mode='binary'`; default)
    * Confusion matrix (`mode='confusionmat'`)

    Parameters
    ----------
    A, B:
        If `mode` is 'binary' (default):

            A: array, shape = [n_samples]
                True values, interpreted as strictly positive or not
                (i.e. converted to binary).

            B: array, shape = [n_samples]
                Predicted values, interpreted as strictly positive or not
                (i.e. converted to binary).

        if `mode` is 'confusionmat':

            A: array-like, shape = [n_classes (true), n_classes (pred)]
                Confusion matrix, where the element M_{rc} means
                the number of times when the classifier or subject
                guesses that a test sample in the r-th class
                belongs to the c-th class.

            B: ignored

    balanced: bool, optional (default=False)
        Returns the balanced accuracy (equal weight for positive and
        negative values).

    collation: None or array-like of shape = [n_groupings,
        n_classes], optional (default=None)
        Defines how to group entries in `M` to make sub-confusion matrices
        when `mode` is 'confusionmat'.  See `confusion_matrix_stats()`
        for details.

    Returns
    -------
    acc: float or array of shape = [n_groupings]
        An accuracy score (zero-one score) or array of accuracies,
        where each element corresponds to each grouping of
        positives and negatives (when `mode` is 'confusionmat').

    References
    ----------
    http://en.wikipedia.org/wiki/Accuracy
    """

    if mode == 'binary':
        y_true, y_pred = A, B
        assert len(y_true) == len(y_pred)
        assert np.isfinite(y_true).all()
        assert np.isfinite(y_pred).all()

        # -- "binarize" the arguments
        y_true = np.array(y_true) > 0
        assert y_true.ndim == 1

        y_pred = np.array(y_pred) > 0
        assert y_pred.ndim == 1

        i_pos = y_true > 0
        i_neg = ~i_pos

        P = float(i_pos.sum())
        N = float(i_neg.sum())
        TP = float((y_true[i_pos] == y_pred[i_pos]).sum())
        TN = float((y_true[i_neg] == y_pred[i_neg]).sum())

    elif mode == 'confusionmat':
        # A: confusion mat
        # row means true classes, col means predicted classes
        P, N, TP, TN, _, _ = confusion_matrix_stats(A, \
                collation=collation, fudge_mode='none')

    else:
        raise ValueError('Invalid mode')

    if balanced:
        sensitivity = TP / P
        specificity = TN / N
        acc = (sensitivity + specificity) / 2.
    else:
        acc = (TP + TN) / (P + N)

    return acc
