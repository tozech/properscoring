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

@guvectorize(["void(float64[:], float64[:,:], float64[:], float64[:], float64[:])"], 
             "(n),(n,m)->(),(),()", nopython=True)
def _mean_crps_rel_crps_pot(observations, forecasts, mean_crps, reliability, crps_pot):
    mea = 0 #mean CRPS over all observations i
    rel = 0 #mean realibility component
    pot = 0 #CRPS_pot component for computation of resolution component
    
    M = forecasts.shape[1]
    N = forecasts.shape[0]
    for k in range(M+1):
        p_k = k / M
        print(p_k)
        alpha_k = 0
        beta_k = 0
        for i in range(N):
            e = forecasts[i, :]
            x = observations[i]
            if (k > 0) and (k < M):
                if x > e[k+1]:
                    alpha_k += e[k+1] - e[k]
                    beta_k += 0
                elif (e[k+1] >= x) and (x > e[k]):
                    alpha_k += x - e[k]
                    beta_k += e[k+1] - x
                elif e[k] >= x:
                    alpha_k += 0
                    beta_k += e[k+1] - e[k]
            elif (k == 0) or (k == M):
                if x < e[0]:
                    alpha_k += 0
                    beta_k += e[0] - x
                if x > e[M]:
                    alpha_k += x - e[M] #TODO check index k
                    beta_k += 0

        print('k', k, 'Alpha', alpha_k, 'Beta', beta_k)
    mean_crps[0] = 0.
    reliability[0] = 1.
    crps_pot[0] = 3.


if __name__ == '__main__':
#    unittest.main()
    obs = np.array([2, -2, 101])
    fc = np.array([[0, 1, 3, 5], [-1, 0, 6, 7], [9, 10, 11, 100]])
    m, r, p = _mean_crps_rel_crps_pot(obs, fc)
    