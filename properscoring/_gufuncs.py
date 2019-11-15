import numpy as np
from numba import guvectorize


@guvectorize(["void(float64[:], float64[:], float64[:], float64[:])"],
             "(),(n),(n)->()", nopython=True)
def _crps_ensemble_gufunc(observation, forecasts, weights, result):
    # beware: forecasts are assumed sorted in NumPy's sort order

    # we index the 0th element to get the scalar value from this 0d array:
    # http://numba.pydata.org/numba-doc/0.18.2/user/vectorize.html#the-guvectorize-decorator
    obs = observation[0]

    if np.isnan(obs):
        result[0] = np.nan
        return

    total_weight = 0.0
    for n, weight in enumerate(weights):
        if np.isnan(forecasts[n]):
            # NumPy sorts NaN to the end
            break
        if not weight >= 0:
            # this catches NaN weights
            result[0] = np.nan
            return
        total_weight += weight

    obs_cdf = 0
    forecast_cdf = 0
    prev_forecast = 0
    integral = 0

    for n, forecast in enumerate(forecasts):
        if np.isnan(forecast):
            # NumPy sorts NaN to the end
            if n == 0:
                integral = np.nan
            # reset for the sake of the conditional below
            forecast = prev_forecast
            break

        if obs_cdf == 0 and obs < forecast:
            integral += (obs - prev_forecast) * forecast_cdf ** 2
            integral += (forecast - obs) * (forecast_cdf - 1) ** 2
            obs_cdf = 1
        else:
            integral += ((forecast - prev_forecast)
                         * (forecast_cdf - obs_cdf) ** 2)

        forecast_cdf += weights[n] / total_weight
        prev_forecast = forecast

    if obs_cdf == 0:
        # forecast can be undefined here if the loop body is never executed
        # (because forecasts have size 0), but don't worry about that because
        # we want to raise an error in that case, anyways
        integral += obs - forecast

    result[0] = integral


@guvectorize(["void(float64[:], float64[:], float64[:], float64[:])"],
             "(),(n),(m)->(m)", nopython=True)
def _threshold_brier_score_gufunc(observation, forecasts, thresholds, result):
    # both forecasts and thresholds are assumed sorted in NumPy's sort order
    obs = observation[0]

    n_thresholds = len(thresholds)
    n_forecasts = len(forecasts)
    while np.isnan(forecasts[n_forecasts - 1]) and n_forecasts > 0:
        n_forecasts -= 1

    if np.isnan(obs) or n_forecasts == 0:
        result[:] = np.nan
        return

    inv_n_forecasts = 1.0 / n_forecasts

    i = 0
    j = 0
    while i < n_forecasts and j < n_thresholds:
        forecast = forecasts[i]
        threshold = thresholds[j]

        if forecast <= threshold:
            i += 1
        else:
            probability = i * inv_n_forecasts
            binary_obs = obs <= threshold
            result[j] = (probability - binary_obs) ** 2
            j += 1

    for k in range(j, n_thresholds):
        threshold = thresholds[k]
        binary_obs = obs <= threshold
        # probability is always 1, so we can skip the square
        result[k] = 1 - binary_obs

@guvectorize(["void(float64[:], float64[:])"], "(n)->()", nopython=True)
def _uncertainty_comp(observations, result):
    unc = 0.
    N = observations.shape[0]
    for i in range(N):
        for j in range(i):
            unc += np.abs(observations[i] - observations[j])

    result[0] = unc / N**2

@guvectorize(["void(float64[:], float64[:,:], float64[:], float64[:], float64[:])"],
             "(n),(n,m)->(),(),()", nopython=True)
def _mean_crps_rel_pot(observations, forecasts, mean_crps, reliability, crps_pot):
    # beware: forecasts are assumed sorted in NumPy's sort order
    mea = 0 #mean CRPS over all observations i
    rel = 0 #mean realibility component
    pot = 0 #CRPS_pot component for computation of resolution component

    M = forecasts.shape[1]
    N = forecasts.shape[0]
    for k in range(M+1):
        p_k = k / M
        alpha_k = 0
        beta_k = 0
        for i in range(N):
            e = forecasts[i, :]
            x = observations[i]
            print('i', i, 'e', e, 'x', x, 'Alpha', alpha_k, 'Beta', beta_k)
            if (k > 0) and (k < M):
                if e[k] < x:
                    print('case 1')
                    alpha_k += e[k] - e[k-1]
                    beta_k += 0
                elif (e[k-1] < x) and (x <= e[k]):
                    print('case 2')
                    alpha_k += x - e[k-1]
                    beta_k += e[k] - x
                elif x <= e[k-1]:
                    print('case 3')
                    alpha_k += 0
                    beta_k += e[k] - e[k-1]
            elif (k == 0):
                if x < e[0]:
                    print('case 4')
                    alpha_k += 0
                    beta_k += e[0] - x
            elif (k == M):
                if e[M-1] < x:
                    print('case 5')
                    alpha_k += x - e[M-1]
                    beta_k += 0
        alpha_k /= N
        beta_k /= N

        print('k', k, 'p_k', p_k, 'Alpha', alpha_k, 'Beta', beta_k)
        mea += alpha_k * p_k**2 + beta_k * (1 - p_k)**2

        g_k = alpha_k + beta_k
        if g_k > 0:
            o_k = beta_k / g_k

            rel += g_k * (o_k - p_k)**2
            pot += g_k * o_k * (1 - o_k)

    mean_crps[0] = mea
    reliability[0] = rel
    crps_pot[0] = pot
