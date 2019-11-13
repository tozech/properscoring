#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:36:45 2019

@author: tzech
"""
import unittest

import numpy as np

from properscoring._mean_crps import _mean_crps_hersbach

class TestsEnsembleComponents(unittest.TestCase):
    def setUp(self):
        self.obs = np.random.randn(10)
        self.forecasts = np.random.randn(10, 5)
        
    def test_crps_rel_res_unc(self):
        res = _mean_crps_hersbach(self.obs, self.forecasts)
        self.assertAlmostEqual(res[0], 0)