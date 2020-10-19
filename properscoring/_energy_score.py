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
    weights : np.ndarray
        1-dim (samples, members)

    Returns
    -------
    np.ndarray
        1-dim (samples) energy score

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
    #deltas = fc_extra - fc_extra.transpose((-1, -2))


#    observations = np.asarray(observations)
#    forecasts = np.asarray(forecasts)
#    if axis != -1:
#        forecasts = move_axis_to_end(forecasts, axis)
#
#    if weights is not None:
#        weights = move_axis_to_end(weights, axis)
#        if weights.shape != forecasts.shape:
#            raise ValueError('forecasts and weights must have the same shape')
#
#    if observations.shape not in [forecasts.shape, forecasts.shape[:-1]]:
#        raise ValueError('observations and forecasts must have matching '
#                         'shapes or matching shapes except along `axis=%s`'
#                         % axis)
#
#    if observations.shape == forecasts.shape:
#        if weights is not None:
#            raise ValueError('cannot supply weights unless you also supply '
#                             'an ensemble forecast')
#        return abs(observations - forecasts)
#
#    if not issorted:
#        if weights is None:
#            forecasts = np.sort(forecasts, axis=-1)
#        else:
#            idx = argsort_indices(forecasts, axis=-1)
#            forecasts = forecasts[idx]
#            weights = weights[idx]
#
#    if weights is None:
#        weights = np.ones_like(forecasts)
#
#    return _energy_score_gufunc(observations, forecasts)