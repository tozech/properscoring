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
        self.assertAlmostEqual(es, 0.)
        
    def test_one_observation_diff_obs(self):
        obs = np.array([[1, 3]])
        fc = np.array([[[1, 2], [1, 2]]])
        es = energy_score(obs, fc)
        print(es)
        self.assertEqual(es.shape, (1,))
        self.assertAlmostEqual(es[0], 1.)

