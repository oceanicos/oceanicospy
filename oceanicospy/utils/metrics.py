import numpy as np

def stats(x, y):
    """Compute R (Pearson), Bias (model-obs), RMSE, and Scatter Index."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(x, y)[0, 1]
    bias = float(np.nanmean(y - x))
    rmse = float(np.sqrt(np.nanmean((y - x) ** 2)))
    si = rmse / float(np.nanmean(x)) if np.nanmean(x) != 0 else np.nan
    return r, bias, rmse, si