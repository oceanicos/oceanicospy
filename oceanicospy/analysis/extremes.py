import numpy as np
from scipy.stats import genpareto

def POT_method(annual_exceedances, vble_series, threshold, num_years):
    """
    Peaks Over Threshold (POT) analysis.

    Parameters
    ----------
    annual_exceedances : array-like, list
        Array of annual exceedances.
    vble_series : array-like
        Variable series data.
    threshold : float
        Threshold value for POT analysis.
    num_years : int
        Number of years of data.

    Returns
    -------
    return_levels : array-like
        Generalized Pareto distribution quantiles for the given return periods.
    """
    annual_exceedances = np.array(annual_exceedances)
    x = vble_series - threshold
    if np.all(vble_series[:] == 0):
        return_levels = np.ones(annual_exceedances.shape) * np.nan
    else:
        over_threshold = x[x > 0]
        shape, loc, scale_fit = genpareto.fit(over_threshold, floc=0)
        Npot = len(over_threshold)
        return_periods = num_years / (Npot * (1 / annual_exceedances))
        return_levels = genpareto.isf(return_periods, c=shape, loc=threshold, scale=scale_fit)
    return return_levels

