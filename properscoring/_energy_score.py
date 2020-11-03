#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:16:18 2020

@author: tzech
"""
import numpy as np

from ._utils import suppress_warnings

try:
    from properscoring._gufuncs import _energy_score_gufunc
except ImportError as exc:
    def _make_import_error(a):
        raise ImportError('Numba is not installed.')
    _energy_score_gufunc = lambda x: _make_import_error(x)

# TODO: refactor energy_score to energy_score_vectorized and add numba version

def energy_score(observations, forecasts, weights=None, issorted=False,
                 axis=-2, feature_axis=-1):
    """computes the energy score

    Parameters
    ----------
    observations : np.ndarray
        2-dim (samples, features)
    forecasts : np.ndarray
        3-dim (samples, members, features)
    weights : np.ndarray, optional
        2-dim (samples, members)
    issorted : bool, optional
    axis : int, optional
    feature_axis : int, optional

    Returns
    -------
    np.ndarray
        1-dim (samples) energy score

    References
    ----------
    Tilmann Gneiting & Adrian E Raftery (2007) Strictly Proper Scoring Rules,
    Prediction, and Estimation, Journal of the American Statistical Association,
    102:477, 359-378, DOI: 10.1198/016214506000001437
    """
    if issorted:
        raise NotImplementedError
    if axis != -2:
        raise NotImplementedError
    if feature_axis != -1:
        raise NotImplementedError

    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    weights = np.asarray(weights)
    if weights.ndim > 0:
        forecasts_nan = np.all(~np.isnan(forecasts), axis=-1)
        weights = np.where(forecasts_nan, weights, np.nan)
        #Uses mean for NaN handling, requires mean in score = np.nanmean(... later on
        weights = weights / np.nanmean(weights, axis=-1, keepdims=True)
    else:
        weights = np.ones(forecasts.shape[:-1])
        weights = weights / np.nanmean(weights, axis=-1, keepdims=True)

    if observations.ndim == forecasts.ndim - 1:
        # sum over the last axis
#        assert observations.shape == forecasts.shape[:-1] #TODO redo
        observations = np.expand_dims(observations, axis=-2)
        l2norm_resi = np.linalg.norm(forecasts - observations, axis=feature_axis)
        with suppress_warnings('Mean of empty slice'):
            score = np.nanmean(weights * l2norm_resi, axis=-1)
        # insert new axes along last and second to last forecast dimensions so
        # forecasts_diff expands with the array broadcasting
        forecasts_diff = (np.expand_dims(forecasts, -2) -
                          np.expand_dims(forecasts, -3))
        weights_matrix = (np.expand_dims(weights, -1) *
                          np.expand_dims(weights, -2))
        l2norm_diff = np.linalg.norm(forecasts_diff, axis=feature_axis)
        with suppress_warnings('Mean of empty slice'):
            score += -0.5 * np.nanmean(weights_matrix * l2norm_diff, axis=(-2, -1))

        return score
