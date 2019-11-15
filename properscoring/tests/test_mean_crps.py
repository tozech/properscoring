#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:36:45 2019

@author: tzech
"""
import unittest

import numpy as np
from numba import guvectorize

from properscoring._gufuncs import _uncertainty_comp, _mean_crps_rel_pot
from properscoring._mean_crps import _mean_crps_hersbach
#from properscoring import crps_ensemble

#
class TestsEnsembleComponents(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.obs = np.random.randn(20)
        self.forecasts = np.random.randn(20, 5)

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

    def test_mean_crps_rel_pot_basic(self):
        obs = np.array([2, -2, 13])
        fc = np.array([[0, 1, 3, 5], [-1, 0, 6, 7], [2, 9, 10, 11]])
        m, r, p = _mean_crps_rel_pot(obs, fc)
        self.assertAlmostEqual(m, 2.3541666666666665)# Equals crps_ensemble(obs, fc).mean()
        self.assertAlmostEqual(r, 1.2893518518518519)
        self.assertAlmostEqual(p, 1.0648148148148149)

    def test_mean_crps(self):
        m, rel, res, unc = _mean_crps_hersbach(self.obs, self.forecasts)
#        self.assertAlmostEqual(m, rel - res + unc)
        self.assertAlmostEqual(m, 0.6140103173775591)# Equals crps_ensemble(obs, fc).mean()
        self.assertAlmostEqual(rel, 0.23936183131677677)
        self.assertAlmostEqual(res, 0.15370836411500766)
        self.assertAlmostEqual(unc, 0.5283568501757899)

    def test_mean_crps_with_g_k_yields_zero(self):
        # If g_k is zero for some k, then o_k does not exists
        # By accident, this is the case for this normal dist with n=10, seed=42
        np.random.seed(42)
        obs = np.random.randn(10)
        forecasts = np.random.randn(10, 5)
        m, rel, res, unc = _mean_crps_hersbach(obs, forecasts)
        self.assertFalse(np.isnan(rel))
        self.assertFalse(np.isnan(res))
        self.assertAlmostEqual(m, 0.6973680827091615)
        self.assertAlmostEqual(rel, 0.3418658027844771)
        self.assertAlmostEqual(res, 0.025998022773329965)
        self.assertAlmostEqual(unc, 0.38150030269801427)


if __name__ == '__main__':
    unittest.main()
    #%%
#    obs = np.array([2, -2, 13])
#    fc = np.array([[0, 1, 3, 5], [-1, 0, 6, 7], [2, 9, 10, 11]])
#    m, r, p = _mean_crps_rel_pot(obs, fc)
