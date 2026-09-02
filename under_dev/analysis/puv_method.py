import numpy as np
import pandas as pd
import time

from scipy.signal import detrend
from scipy.signal import welch, windows
from PyEMD import CEEMDAN,EMD

from ..utils import wave_props,constants,extras

np.seterr(divide = 'ignore') 

def crosgk(X, Y, N, M, DT=1, DW=1, stats=0):
    """
    Power cross-spectrum computation, with smoothing in the frequency domain
    
    Parameters
    ----------
    X : list or ndarray
        series 1
    Y : list or ndarray
        series 2
    N : list or ndarray
        number of samples per data segment (power of 2)
    M : list or ndarray
        number of frequency bins over which is smoothed (optional), 
        no smoothing for M=1 (default)
    DT : float
        Time step (optional), default DT=1
    DW : int
        Data window type (optional): DW = 1 for Hann window (default)
                                     DW = 2 for rectangular window
    stats :  bool
        Display resolution, degrees of freedom (optimal, YES=1, NO=0)

    Returns
    -------
    P : ndarray
        Contains the (cross-)spectral estimates: column 0 = Pxx, 1 = Pyy, 2 = Pxy
    F : ndarray
        Contains the frequencies at which P is given

    Notes
    -----
    This script is adapted from the matlab script made by Gert Klopman, Delft Hydraulics, 1995
    """
  
    df = 1 / (N * DT)

    # data window
    if DW == 1:
        # Hann
        w = np.hanning(N)
        dj = N / 2
    else:
        # rectangle
        w = np.ones(N)
        dj = N
    varw = np.sum(w**2) / N

    # summation over segments
    nx = max(X.shape)
    ny = max(Y.shape)
    avgx = np.sum(X) / nx
    avgy = np.sum(Y) / ny
    px = np.zeros(N)
    py = np.zeros(N)
    Pxx = np.zeros(N)
    Pxy = np.zeros(N)
    Pyy = np.zeros(N)
    ns = 0

    for j in range(0, nx - N + 1, int(dj)):
        ns += 1

        # compute FFT of signals
        px = X[j:j + N] - avgx
        px = w * px
        px = np.fft.fft(px)

        py = Y[j:j + N] - avgy
        py = w * py
        py = np.fft.fft(py)

        # compute periodogram
        Pxx = Pxx + px * np.conj(px)
        Pyy = Pyy + py * np.conj(py)
        Pxy = Pxy + py * np.conj(px)

    Pxx = (2 / (ns * (N**2) * varw * df)) * Pxx
    Pyy = (2 / (ns * (N**2) * varw * df)) * Pyy
    Pxy = (2 / (ns * (N**2) * varw * df)) * Pxy

    # smoothing
    if M > 1:
        w = np.hamming(M)
        w = w / np.sum(w)
        w = np.concatenate((w[int(np.ceil((M + 1) / 2)) - 1:M], np.zeros(N - M), w[0:int(np.ceil((M + 1) / 2)) - 1]))
        w = np.fft.fft(w)
        Pxx = np.fft.fft(Pxx)
        Pyy = np.fft.fft(Pyy)
        Pxy = np.fft.fft(Pxy)
        Pxx = np.fft.ifft(w * Pxx)
        Pyy = np.fft.ifft(w * Pyy)
        Pxy = np.fft.ifft(w * Pxy)

    Pxx = np.real(Pxx[:N // 2])
    Pyy = np.real(Pyy[:N // 2])
    Pxy = np.real(Pxy[:N // 2])

    # frequencies
    F = np.arange(0, N // 2 + 1) * df

    # signal variance
    if DW == 1:
        nn = (ns + 1) * N / 2
    else:
        nn = ns * N
    avgx = np.sum(X[:int(nn)]) / nn
    varx = np.sum((X[:int(nn)] - avgx) ** 2) / (nn - 1)
    avgy = np.sum(Y[:int(nn)]) / nn
    vary = np.sum((Y[:int(nn)] - avgy) ** 2) / (nn - 1)
    covxy = np.sum((X[:int(nn)] - avgx) * (Y[:int(nn)] - avgy)) / (nn - 1)

    m0xx = (0.5 * Pxx[0] + np.sum(Pxx[1:N // 2 - 1]) + 0.5 * Pxx[N // 2 - 1]) * df
    m0yy = (0.5 * Pyy[0] + np.sum(Pyy[1:N // 2 - 1]) + 0.5 * Pyy[N // 2 - 1]) * df
    m0xy = (0.5 * Pxy[0] + np.sum(Pxy[1:N // 2 - 1]) + 0.5 * Pxy[N // 2 - 1]) * df

    Pxx = Pxx * (varx / m0xx)
    Pyy = Pyy * (vary / m0yy)
    Pxy = Pxy * (covxy / np.real(m0xy))

    P = np.column_stack((Pxx, Pyy, Pxy))

    # output spectrum characteristics
    dof = np.floor(2 * ns * (M + 1) / 2 / (3 - DW))
    if stats == 1:
        print(f'number of samples used : {int(nn):8d}')
        print(f'degrees of freedom     : {int(dof):8d}')
        print(f'resolution             : {((3 - DW) * df * (M + 1) / 2):13.5f}')

    return P, F, dof

@extras.timing_decorator
def compute_spectrum_from_puv(p,u,v,sampling_freq,anchoring_depth,sensor_height):
    """
    This function uses the PUV method to obtain the frequency spectrum of the waves measured
    with a pressure sensor at a specified depth
    
    The scalar spectrum is estimated using the Fast Fourier Transform by windows to reduce
    the lekaege. 
    
    Parameters
    ----------
    p : list or ndarray
        An array of the detrended presure records per burst
    u : list or ndarray
        An array of the detrended x-component of velocity records per burst
    v : list or ndarray
        An array of the detrended y-component of velocity presure records per burst   
    sampling_freq : float
        Sampling frequency for the records
    anchoring_depth : float
        Depth at where the device was settled on the bottom.
    sensor_height: float
        Distance from the sensor to the bottom.        
    
    Returns
    -------
    S : ndarray
        Density variance spectrum
    Dir : ndarray
        Direction array    
    f: ndarray
        Frequency of the spectrum    
    Su: ndarray
        x-component of velocity
    Sv: ndarray
        y-component of velocity

    Notes
    -----
    The SWASH (Simulating WAves till SHore) model script: crosgk.m is used to estimate the cross-spectrum
    between pressure velocity u and velocity v.

    18-Jan-2014 : First Matlab function - Daniel Peláez

    """

    # Variable definition
    nfft = 128
    f = (sampling_freq / nfft) * np.arange(0, nfft / 2)
    f = f.reshape(-1, 1)

    # Low frequencies cutoff
    cutoff = 0.03
    ix = np.where(f >= cutoff)[0]
    f = f[ix]

    # Dispersion relation
    w = 2 * np.pi * f
    k0 = (w**2) / constants.GRAVITY
    for cnt in range(100):
        k = (w**2) / (constants.GRAVITY * np.tanh(k0 * anchoring_depth))
        k0 = k

    # Transference function
    Kp = np.cosh(k * anchoring_depth) / np.cosh(k * sensor_height)
    Kp[Kp > 10] = 10

    # Cross-spectrum
    Pp, _, _ = crosgk(p, p, nfft, 1, 1/sampling_freq, 1, 0)
    Pu, _, _ = crosgk(p, u, nfft, 1, 1/sampling_freq, 1, 0)
    Pv, _, _ = crosgk(p, v, nfft, 1, 1/sampling_freq, 1, 0)

    # Normalization factor
    fac = 1

    # Pressure
    Sp = Pp[:, 0]
    Sp = Sp / fac

    # Velocity
    Su = Pu[:, 2]
    Su = Su[ix] / fac
    Sv = Pv[:, 2]
    Sv = Sv[ix] / fac

    Kp=Kp.flatten()
    S = Sp[ix] * (Kp**2)

    # Direction
    Dir = (180 / np.pi) * np.arctan2(Sv, Su)
    Dir[Dir < 0] += 180

    return S, Dir, f, Su, Sv

def spectra_from_puv(clean_records,sampling_data):
    """
    Compute wave spectra and wave parameters from pressure, u, and v components.

    Parameters
    ----------
    clean_records : pandas.DataFrame
        DataFrame containing cleaned records with columns for 'pressure', 'u', 'v', and 'burstId'.
    sampling_data : dict
        Dictionary containing sampling information with keys:
        - 'sampling_freq': Sampling frequency of the data.
        - 'anchoring_depth': Depth at which the sensor is anchored.
        - 'sensor_height': Height of the sensor above the seabed.

    Returns
    -------
    wave_spectra_data : dict
        Dictionary containing wave spectra data with keys:
        - 'S': List of power spectra for each burst.
        - 'dir': List of direction spectra for each burst.
        - 'freq': List of frequency arrays for each burst.
        - 'time': List of timestamps corresponding to each burst.
    wave_params_data : pandas.DataFrame
        DataFrame containing wave parameters with columns:
        - 'time': Timestamps corresponding to each burst.
        - 'Hm0': Zero-moment wave height.
        - 'Hrms': Root mean square wave height.
        - 'Hmean': Mean wave height.
        - 'Tp': Peak period.
        - 'Tm01': Mean period (first moment).
        - 'Tm02': Mean period (second moment).
    """
    wave_params=["time","Hm0","Hrms","Hmean","Tp","Tm01","Tm02"]
    wave_params_data={param:[] for param in wave_params}

    wave_spectra_vars=["S","dir","freq","time"]
    wave_spectra_data={var:[] for var in wave_spectra_vars}

    clean_data=clean_records

    for i in clean_data['burstId'].unique():

        burst_series = clean_data[clean_data['burstId'] == i]

        burst_series_detrended = burst_series.iloc[:,:-1].apply(lambda x: detrend(x, type='constant'), axis=0)
        burst_series_detrended[clean_records.columns[-1]] = burst_series.iloc[:, -1]

        # Compute the spectrum
        power, direction, freqs, Su, Sv = spectrum_puv_method(burst_series_detrended['pressure'],burst_series_detrended['u'],
                                                                        burst_series_detrended['v'],sampling_data['sampling_freq'], 
                                                                        sampling_data['anchoring_depth'], sampling_data['sensor_height'])                                                     
        wave_spectra_data["S"].append(power)
        wave_spectra_data["dir"].append(direction)
        wave_spectra_data["freq"].append(freqs)
        wave_spectra_data["time"].append(burst_series_detrended.index[0])

        # Compute wave parameters from the spectrum
        Hm0, Hrms, Hmean, Tp, Tm01, Tm02 = wave_params_from_spectrum_v1(power, freqs)

        # Store wave parameters
        wave_params_data['time'].append(burst_series_detrended.index[0])
        wave_params_data['Hm0'].append(Hm0)
        wave_params_data['Hrms'].append(Hrms)
        wave_params_data['Hmean'].append(Hmean)
        wave_params_data['Tp'].append(Tp)
        wave_params_data['Tm01'].append(Tm01)
        wave_params_data['Tm02'].append(Tm02)

    wave_params_data = pd.DataFrame(wave_params_data).set_index('time')
    return wave_spectra_data,wave_params_data

