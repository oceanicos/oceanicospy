import pandas as pd
import utide

def tide_harmonic_decomposition(data, time, lat, trend=False, method='ols', conf_int='MC', verbose=False):
    """
    Perform tidal harmonic analysis on a sea level time series using UTide.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        1D array or Series containing sea level observations.
    time : array-like
        Time vector associated with the sea level data.
    lat : float
        Latitude of the measurement site, required by UTide to compute
        the Coriolis frequency used in tidal constituent estimation.
    trend : bool, optional
        If True, a linear trend is included in the harmonic fit.
        Default is False.
    method : str, optional
        Regression method for the harmonic fit. Accepted values are
        ``'ols'`` (ordinary least squares) and ``'robust'``.
        Default is ``'ols'``.
    conf_int : str, optional
        Method used to estimate confidence intervals. Accepted values
        are ``'MC'`` (Monte Carlo), ``'linearized'``, and ``'none'``.
        Default is ``'MC'``.
    verbose : bool, optional
        If True, prints the harmonic analysis coefficients to stdout.
        Default is False.

    Returns
    -------
    serieTable : pd.DataFrame
        DataFrame containing the following columns:

        - ``time`` : original time vector.
        - ``sea_level`` : observed sea level.
        - ``tide`` : reconstructed tidal signal from harmonic constituents.
        - ``residual`` : non-tidal residual computed as observed minus tide.

    coefTable : pd.DataFrame
        DataFrame with one row per tidal constituent, containing:

        - ``name`` : constituent name (e.g., M2, S2, K1).
        - ``Amplitude`` : fitted amplitude.
        - ``A_ci`` : 95% confidence interval for the amplitude.
        - ``Phase`` : fitted phase in degrees.
        - ``P_ci`` : 95% confidence interval for the phase.
        - ``PE`` : percentage of total variance explained by each constituent.
        - ``SNR`` : signal-to-noise ratio (values > 2 indicate significance).

    Raises
    ------
    ValueError
        If ``data`` is not a 1D array or pandas Series.

    Notes
    -----
    The residual signal (observed - tide) captures non-tidal contributions
    such as storm surges, seiches, and long-term sea level trends.
    Constituents with SNR < 2 should be interpreted with caution.

    **Development history**
    
    - 01-Apr-2026 : Juan Diego Toro


    Examples
    --------
    >>> serieTable, coefTable = TideHarmonics(data=sl, time=t, lat=-6.2)
    >>> coefTable[coefTable['SNR'] > 2]  # filter significant constituents
    """

    # Validate that the input data is one-dimensional
    if data.ndim == 1:

        # Build a DataFrame depending on the type of the input data
        if isinstance(data, pd.Series):
            # pandas Series: use the provided time vector as the index directly
            dataset = pd.DataFrame(data, columns=['data'], index=time)
        else:
            # numpy array: convert time to DatetimeIndex before indexing
            time_index = pd.to_datetime(time)
            dataset = pd.DataFrame(data, columns=['data'], index=time_index)

        # Decompose the sea level series into tidal constituents (e.g., M2, S2,
        # K1, O1) by solving the harmonic regression with the specified method.
        coef = utide.solve(dataset.index.values, dataset['data'].values, lat=lat, trend=trend, method=method, conf_int=conf_int)

        # Optionally print the fitted coefficients for inspection
        if verbose:
            print("Harmonic Analysis Coefficients:")
            print(coef)

        # Reconstruct the tidal signal from the fitted harmonic coefficients
        tide = utide.reconstruct(dataset.index.values, coef, verbose=verbose)

        # Assemble the time series table with observed, tidal, and residual signals
        serieTable = pd.DataFrame({
            'time': dataset.index.values,
            'sea_level': dataset['data'].values,
            'tide': tide.h,
            'residual': dataset['data'].values - tide.h  # non-tidal residual
        })

        # Assemble the constituent table with amplitudes, phases, and fit statistics
        coefTable = pd.DataFrame({
            'name': coef.name,
            'Amplitude': coef.A,
            'A_ci': coef.A_ci,    # 95% confidence interval on amplitude
            'Phase': coef.g,
            'P_ci': coef.g_ci,    # 95% confidence interval on phase
            'PE': coef.PE,        # percentage of variance explained
            'SNR': coef.SNR       # signal-to-noise ratio
        })

        return serieTable, coefTable

    else:
        # Raise an error if the input data has more than one dimension
        raise ValueError('The input data must be a 1D np array or a pandas Series.')