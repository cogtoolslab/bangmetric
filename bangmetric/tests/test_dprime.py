"""Test suite for the ``dprime`` module"""

import numpy as np
from pytest import raises

from bangmetric import dprime

ATOL = 1e-6


def test_error_length():
    y_pred = np.array([False, True, True])
    y_true = np.array([True, False])
    raises(AssertionError, dprime, y_pred, y_true)


def test_basic():
    y_true = np.array([False, True, True, True, False, False, False, True])
    y_pred = np.array([0.491, -0.1, 0.64, 1.52, -0.23, -0.23, 1.579, 0.76])
    dp = dprime(y_true, y_pred)
    reference = 0.39541092958803298
    assert abs(dp - reference) < ATOL


def test_basic100():
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.5, size=100)
    y_pred = rng.randn(y_true.size)
    dp = dprime(y_true, y_pred)
    reference = -0.20652941441924857
    assert abs(dp - reference) < ATOL


def test_dprime_undefined():
    y_true = np.ones((5), dtype=bool)
    y_pred = np.random.randn(y_true.size)
    y_pred -= y_pred.min()
    y_pred += 1.

    dp_pos = dprime(y_true, y_pred)
    assert dp_pos is None

    dp_neg = dprime(-y_true, -y_pred)
    assert dp_neg is None


def test_error_non_finite():
    y_true = np.zeros((5), dtype=float)
    y_pred = np.ones_like(y_true)
    y_pred[0] = np.nan
    raises(AssertionError, dprime, y_true, y_pred)
    y_pred[0] = np.inf
    raises(AssertionError, dprime, y_true, y_pred)
