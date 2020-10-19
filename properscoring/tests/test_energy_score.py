import functools
import unittest
import warnings

import numpy as np
from scipy import stats, special
from numpy.testing import assert_allclose

from properscoring import energy_score


class TestESSimple(unittest.TestCase):
    def test_one_observation_trivial(self):
        obs = np.array([[1, 2]])
        fc = np.array([[[1, 2], [1, 2]]])
        es = energy_score(obs, fc)
        self.assertAlmostEqual(es[0], 0.)

    def test_one_observation_diff_obs(self):
        obs = np.array([[1, 3]])
        fc = np.array([[[1, 2], [1, 2]]])
        es = energy_score(obs, fc)
        self.assertEqual(es.shape, (1,))
        self.assertAlmostEqual(es[0], 1.)

    def test_one_ensemble_diff(self):
        obs = np.array([[1, 2]])
        fc = np.array([[[1, 2], [0, 2]]])
        es = energy_score(obs, fc)
        self.assertEqual(es.shape, (1,))
        self.assertAlmostEqual(es[0], 0.25)

    def test_all_different(self):
        obs = np.array([[2, 3]])
        fc = np.array([[[0, 0], [-2, -3]]])
        es = energy_score(obs, fc)
        self.assertEqual(es.shape, (1,))
        self.assertAlmostEqual(es[0], 4.5069390943299865)

    def test_trivial_weights(self):
        obs = np.array([[2, 3]])
        fc = np.array([[[0, 0], [-2, -3]]])
        weights = np.array([[0.5, 0.5]])
        es = energy_score(obs, fc, weights)
        self.assertEqual(es.shape, (1,))
        self.assertAlmostEqual(es[0], 4.5069390943299865)

    def test_different_weights(self):
        obs = np.array([[2, 3]])
        fc = np.array([[[0, 0], [-2, -3]]])
        weights = np.array([[0.2, 0.8]])
        es = energy_score(obs, fc, weights)
        self.assertEqual(es.shape, (1,))
        self.assertAlmostEqual(es[0], 5.9131040917609425)

    def test_one_member_nan(self):
        obs = np.array([[1, 2]])
        fc = np.array([[[1, np.nan], [0, 2]]])
        es = energy_score(obs, fc)
        self.assertEqual(es.shape, (1,))
        self.assertAlmostEqual(es[0], 1.)

    def test_one_member_nan_with_weights(self):
        obs = np.array([[3, 4]])
        fc = np.array([[[0, 0], [-2, np.nan]]])
        weights = np.array([[0.2, 0.8]])
        es = energy_score(obs, fc, weights)
        self.assertEqual(es.shape, (1,))
        self.assertAlmostEqual(es[0], 5.)

    def test_all_members_have_nans(self):
        obs = np.array([[1, 2]])
        fc = np.array([[[1, np.nan], [np.nan, 2]]])
        es = energy_score(obs, fc)
        self.assertEqual(es.shape, (1,))
        np.testing.assert_equal(es[0], np.nan)

# TODO: Use test for CRPS

