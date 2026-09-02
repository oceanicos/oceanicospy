import pandas as pd

def compute_annual_cycle(data, time, freq='Monthly', method='mean'):
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
        
        if freq == 'Monthly':
            if method == 'mean':
                 data = dataset.groupby(dataset.index.month)['data'].mean()
                 error = dataset.groupby(dataset.index.month)['data'].std()
                 return pd.DataFrame({'month': data.index, 'mean': data.values, 'std': error.values})
            elif method == 'median':
                 data = dataset.groupby(dataset.index.month)['data'].median()
                 error = dataset.groupby(dataset.index.month)['data'].std()
                 return pd.DataFrame({'month': data.index, 'median': data.values, 'std': error.values})
    
        elif freq == 'Seasonal':
            if method == 'mean':
                data = dataset.groupby((dataset.index.month%12 + 3)//3)['data'].mean()
                error = dataset.groupby((dataset.index.month%12 + 3)//3)['data'].std()
                return pd.DataFrame({'season': data.index, 'mean': data.values, 'std': error.values})
            elif method == 'median':
                data = dataset.groupby((dataset.index.month%12 + 3)//3)['data'].median()
                error = dataset.groupby((dataset.index.month%12 + 3)//3)['data'].std()
                return pd.DataFrame({'season': data.index, 'median': data.values, 'std': error.values})
    else:
        raise ValueError('The input data must be a 1D np array or a pandas Series.')