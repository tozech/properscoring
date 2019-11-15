#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:36:45 2019

@author: tzech
"""
import unittest

import numpy as np
from numba import guvectorize

from properscoring._gufuncs import _uncertainty_comp



from properscoring._mean_crps import _mean_crps_hersbach
#
class TestsEnsembleComponents(unittest.TestCase):
    def setUp(self):
        self.obs = np.random.randn(10)
        self.forecasts = np.random.randn(10, 5)

#    def test_crps_rel_res_unc(self):
#        res = _mean_crps_hersbach(self.obs, self.forecasts)
#        self.assertAlmostEqual(res[0], 0)

    def test_uncertainty_comp_basic(self):
        obs = np.array([3., 4., 10.])
        act = _uncertainty_comp(obs)
        self.assertAlmostEqual(act, 14 / 9)

    def test_uncertainty_comp_nan(self):
        obs = np.array([3., np.nan, 10.])
        act = _uncertainty_comp(obs)
        self.assertTrue(np.isnan(act))


if __name__ == '__main__':
    unittest.main()