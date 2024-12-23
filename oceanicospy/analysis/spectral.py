import numpy as np
import pandas as pd 

from scipy.signal import detrend

from ..utils import wave_props,constants

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


def spectrum_puv_method(p,u,v,sampling_freq,anchoring_depth,sensor_height):
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




def spectra_from_fft(clean_records,sampling_data):
    """
    Compute wave spectra and wave parameters from FFT.

    Parameters
    ----------
    clean_records : pandas.DataFrame
        DataFrame containing the cleaned records with columns such as 'pressure', 'burstId', etc.
    sampling_data : dict
        Dictionary containing sampling information with keys:
        - 'sampling_freq': Sampling frequency of the data.
        - 'anchoring_depth': Depth at which the sensor is anchored.
        - 'sensor_height': Height of the sensor from the seabed.

    Returns
    -------
    wave_spectra_data : dict
        Dictionary containing wave spectra data with keys:
        - 'S': List of power spectral densities for each burst.
        - 'dir': List of directions (currently not computed, placeholder).
        - 'freq': List of frequency arrays for each burst.
        - 'time': List of timestamps corresponding to each burst.
    wave_params_data : pandas.DataFrame
        DataFrame containing wave parameters with columns:
        - 'Hm0': Zero-moment wave height.
        - 'Hrms': Root mean square wave height.
        - 'Hmean': Mean wave height.
        - 'Tp': Peak period.
        - 'Tm01': Mean period (first moment).
        - 'Tm02': Mean period (second moment).
        - Index is the timestamp corresponding to each burst.
    """

    # The depth is computed dividing the pressure by the density and gravity. 
    # The atmospheric pressure is also subtracted and it is converted from bars to pascals.
    clean_data=clean_records.copy()
    if np.all(['depth' not in column.lower() for column in clean_data.columns]):
        clean_data['depth']=((clean_data['pressure']-constants.ATM_PRESSURE_BAR)*10000)/(constants.WATER_DENSITY*constants.GRAVITY)

    # To eliminate the trend of the series, it is grouped by burst and the average prof of each burst is found.       
    columns_with_variables = [col for col in clean_data.columns if col != 'burstId']
    clean_data[columns_with_variables] = clean_data.groupby('burstId')[columns_with_variables].transform(lambda x: x - x.mean())

    # Subtracting the mean depth in each burst
    clean_data['n']=clean_data['depth']

    wave_params=["time","Hm0","Hrms","Hmean","Tp","Tm01","Tm02"]
    wave_params_data={param:[] for param in wave_params}

    wave_spectra_vars=["S","dir","freq","time"]
    wave_spectra_data={var:[] for var in wave_spectra_vars}

    for i in clean_data['burstId'].unique():
        burst_series = clean_data[clean_data['burstId'] == i]

        # Compute the spectrum
        power, power_kp, freqs, T, kpmin, fmax_kp = spectrum_from_surflevel(burst_series['n'][::2], sampling_data['sampling_freq']/2,
                                                                                sampling_data['anchoring_depth'], 
                                                                                sampling_data['sensor_height'])
        wave_spectra_data["S"].append(power_kp)
        wave_spectra_data["time"].append(burst_series.index[0])
        wave_spectra_data["freq"].append(freqs)

        # Compute wave parameters from the spectrum
        Hm0, Hrms, Hmean, Tp, Tm01, Tm02 = wave_params_from_spectrum_v1(power, freqs)

        # Store wave parameters
        wave_params_data['time'].append(burst_series.index[0])
        wave_params_data['Hm0'].append(Hm0)
        wave_params_data['Hrms'].append(Hrms)
        wave_params_data['Hmean'].append(Hmean)
        wave_params_data['Tp'].append(Tp)
        wave_params_data['Tm01'].append(Tm01)
        wave_params_data['Tm02'].append(Tm02)

    wave_params_data=pd.DataFrame(wave_params_data).set_index('time')
    return wave_spectra_data,wave_params_data

def spectrum_from_surflevel(burst_serie,sampling_freq,anchoring_depth,sensor_height):
    """
    Computes the density variance spectrum based on the Fast Fourier transform. 
    
    Parameters
    ----------
    burst_serie : list or ndarray
        An array of the detrended surface level per burst
    sampling_freq : float
        Sampling frequency for the records    
    anchoring_depth : float
        Depth at where the device was settled on the bottom.
    sensor_height: float
        Distance from the sensor to the bottom.        
    
    Returns
    -------
    power : ndarray
        Density variance spectrum    
    power_kp : ndarray
        Density variance spectrum corrected by Kp    
    freq: ndarray
        Frequency of the spectrum

    Notes
    -----
    Based on https://currents.soest.hawaii.edu/ocn_data_analysis/_static/Spectrum.html

    01-Sep-2023 : Origination - Juan Diego Toro

    """

    freq = np.fft.fftfreq(len(burst_serie),sampling_freq)
    T = 1/freq
    fourier = np.fft.fft(burst_serie)

    # Filtering frequencies 
    target_freqs = np.where((np.abs(freq>=0.003))&((np.abs(freq<=1))))[0]
    freq = freq[target_freqs]
    fourier= fourier[target_freqs]

    # Density variance spectrum
    amplitude = np.abs(fourier)
    power = (amplitude**2)*2/len(burst_serie)*(1/sampling_freq)

    # Correction by Kp
    L = [wave_props.wavelength(1/f,anchoring_depth) for f in freq]
    k = [2*np.pi/l for l in L]
    Kp = np.array([np.cosh(ki*(sensor_height))/np.cosh(ki*(anchoring_depth)) for ki in k])
    Kpmin = (np.cosh(np.pi/(anchoring_depth-sensor_height)*sensor_height))/(np.cosh(np.pi/(anchoring_depth-sensor_height)*anchoring_depth))
    Kp[Kp<Kpmin] = Kpmin
    Kp[Kp>10] = 10
    fmax_kp = 1/(2*np.pi)*np.sqrt(9.8*np.pi/(anchoring_depth-sensor_height)*np.tanh(np.pi/(anchoring_depth-sensor_height)*anchoring_depth))
    power_kp = np.array(power)/(np.array(Kp)**2)
    return power,power_kp,freq,T,Kpmin,fmax_kp




def wave_params_from_spectrum_v1(spectral_density,freqs):
    """
    This function computes different wave integral parameters from the spectrum
    
    Parameters
    ----------
    spectral_density : list or ndarray
        Density variance spectrum
    freq : list or ndarray
        Frequencies of the spectrum
    
    Returns
    -------
    Hs : float
        Significant wave heigth [m]
    Hrms : float
        Root-mean squared wave heigth [m]
    Hmean : float
        Mean wave heigth [m]
    Tp : float
        Peak period [s]
    Tm01 : float
        Mean period - fisrt order [s]
    Tm02 : float
        Mean period - second order [s]
    
    Notes
    -----
    10-Dec-2024 : Origination - Franklin Ayala

    """

    m0 = np.trapz(spectral_density, freqs.flatten())
    m1 = np.trapz(freqs.flatten()*spectral_density, freqs.flatten())
    m2 = np.trapz((freqs.flatten()**2)*spectral_density, freqs.flatten())

    i0 = np.trapz(np.abs(spectral_density)**4, freqs.flatten())
    i1 = np.trapz(freqs.flatten() * np.abs(spectral_density)**4, freqs.flatten())

    Hs = 4.004*np.sqrt(m0)
    Hrms = np.sqrt(8*m0)
    Hmean = np.sqrt(2*np.pi*m0)

    # Tp = i0/i1
    Tp = 1/freqs[np.argmax(spectral_density)]
    Tm01 = m0/m1
    Tm02 = np.sqrt(m0/m2)

    return Hs,Hrms,Hmean,Tp,Tm01,Tm02

def wave_params_from_spectrum_v2(spectral_density,freq,fs,bL):
    """
    This function computes different wave integral parameters from the spectrum
    
    Parameters
    ----------
    spectral_density : list or ndarray
        Density variance spectrum
    freq : list or ndarray
        Frequencies of the spectrum
    fs : float
        Sampling frequency
    bL : float
        Burst length
       
    Returns
    -------
    Hs : float
        Significant wave heigth [m]    
    Hrms : float
        Root-mean squared wave heigth [m]
    Hmean : float
        Mean wave heigth [m]
    Tp : float
        Peak period [s]
    Tm01 : float
        Mean period - fisrt order [s]
    Tm02 : float
        Mean period - second order [s]
    """

    df = fs/bL
    suma = 0
    fs = 0
    ffs = 0
    for i in range (len(spectral_density)):
        suma=suma+spectral_density[i]
        fs = fs+(freq[i]*spectral_density[i])
        ffs = ffs + ((freq[i]**2)*spectral_density[i])
    Mo = df*suma
    M1 = fs*df
    M2 = ffs*df
    Hs = 4.004*np.sqrt(Mo)
    Hrms = np.sqrt(8*Mo)
    Hmean = np.sqrt(2*np.pi*Mo)
    in_max = np.where(spectral_density == np.max(spectral_density))[0][0]
    freq_max = freq[in_max]
    Tp = 1/(freq_max)
    Tm01 = Mo/M1
    Tm02 = np.sqrt(Mo/M2)

    return Hs,Hrms,Hmean,Tp,Tm01,Tm02