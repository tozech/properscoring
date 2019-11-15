#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:37:32 2019

@author: tzech
"""
import numpy as np
from scipy.stats import rankdata

from properscoring._gufuncs import _uncertainty_comp, _mean_crps_rel_pot


#def ranks(obs, ensemble):#, mask=):
##    mask=np.bool_(mask)
##
##    obs=obs[mask]
##    ensemble=ensemble[:,mask]
#    
#    combined=np.vstack((obs[np.newaxis],ensemble))
#
#    # print('computing ranks')
#    ranks=np.apply_along_axis(lambda x: rankdata(x,method='min'),0,combined)
#
#    # print('computing ties')
#    ties=np.sum(ranks[0]==ranks[1:], axis=0)
#    ranks=ranks[0]
#    tie=np.unique(ties)
#
#    for i in range(1,len(tie)):
#        index=ranks[ties==tie[i]]
#        # print('randomizing tied ranks for ' + str(len(index)) + ' instances where there is ' + str(tie[i]) + ' tie/s. ' + str(len(tie)-i-1) + ' more to go')
#        ranks[ties==tie[i]]=[np.random.randint(index[j],index[j]+tie[i]+1,tie[i])[0] for j in range(len(index))]
#    return ranks


def _mean_crps_hersbach(observations, forecasts):
    """mean CRPS with reliability, resolution and uncertainty
    
    following the method of Hersbach 2000
    
    Parameters
    ----------
    observations : numpy.array
    forecasts : numpy.array
    
    Returns
    -------
    4-tuple of floats
        mean CRPS, reliability, resolution and uncertainty component of CRPS
    """
    forecasts = np.sort(forecasts, axis=-1)
    uncertainty = _uncertainty_comp(observations)
    mean_crps, reliability, crps_pot = _mean_crps_rel_pot(observations, forecasts)
    resolution = uncertainty - crps_pot
    del forecasts
    return mean_crps, reliability, resolution, uncertainty

if __name__ == '__main__':
    #%%
    np.random.seed(42)
    obs = np.random.randn(20)
    fc = np.random.randn(20, 5)
    m, rel, res, unc = _mean_crps_hersbach(obs, fc)
    from properscoring import crps_ensemble
    crps_values = crps_ensemble(obs, fc)
#    delta_obs = (obs - fc.T).T
#    delta_obs = np.concatenate([delta_obs, np.expand_dims(delta_obs[:, -1], axis=1)], axis=1)
#
#    delta_ens = np.diff(fc)
#    zeros = np.expand_dims(np.zeros(fc.shape[0]), -1)
#    delta_ens = np.concatenate([zeros, delta_ens, zeros], axis=1)
##    pos_delta_obs = np.where(delta_obs > 0, delta_obs, 0)
#    rank_vals = np.expand_dims(ranks(obs, fc.T), -1)
#    possible_ranks = np.arange(1, fc.shape[1] + 2, 1)
#    ind = np.tile(possible_ranks, (fc.shape[0], 1))
#    is_rank = rank_vals == ind
#    above_rank = rank_vals < ind
#    below_rank = rank_vals > ind
#    #%%    
#    alpha = np.where(below_rank, delta_ens, 0)
#    alpha = np.where(is_rank, delta_obs, alpha)
#    
#    beta = np.where(above_rank, delta_ens, 0)
#    beta = np.where(is_rank, -delta_obs, beta)
#    
#    p = possible_ranks / len(possible_ranks)
#    alpha_bar = np.mean(alpha, 0)
#    beta_bar = np.mean(beta, 0)
#    g_bar = alpha_bar + beta_bar
#    o_bar = beta_bar / g_bar
#    rel = np.sum(g_bar * (o_bar - p)**2)
#    unc = _uncertainty_comp(obs)
#    crps_pot = np.sum(g_bar * o_bar * (1 - o_bar))
#    res = unc - crps_pot
    