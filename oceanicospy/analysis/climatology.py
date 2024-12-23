import numpy as np
import pandas as pd

def compute_annual_cycle(data, time):
    """
    Compute the annual cycle of a given time series data.

    Parameters
    ----------
    data : numpy.ndarray or pandas.Series
        The input time series data. Must be a 1D array or Series.
    time : array-like
        The corresponding time values for the data. Must be convertible to datetime.
        
    Returns
    -------
    numpy.ndarray
        The computed annual cycle, with the mean values for each month.
        
    Raises
    ------
    ValueError
        If the input data is not a 1D numpy array or a pandas Series.
    """
    if data.ndim == 1:
        if isinstance(data, pd.Series):
            dataset = pd.DataFrame(data, columns=['data'], index=time)
        else:
            time_index = pd.to_datetime(time)
            dataset = pd.DataFrame(data, columns=['data'], index=time_index)
        
        dataset['month'] = dataset.index.month
        annual_cycle = dataset.groupby('month')['data'].mean() # The column is called 'data'
        return annual_cycle.values
    else:
        raise ValueError('The input data must be a 1D np array or a pandas Series.')