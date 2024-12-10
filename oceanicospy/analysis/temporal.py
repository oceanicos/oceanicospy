import numpy as np
from ..utils import wave_props

def zero_crossing(burst,fs,h,zp):
    """
    This function calculates the significant wave height, the period and the wavelength
    with the zero-crossing method.
    
    Parameters
    ----------
    burst : array_like
        A series of data without trend.
    fs : float
        The sampling frequency.
    h : float
        The measurement depth.
    zp : float
        The distance from the bottom to the sensor.

    Returns
    -------
    Hs : float
        The significant wave height.
    Tm : float
        The mean period.

    Notes
    -----

    23-Feb-2014 : First Matlab version - Daniel Peláez
    01-Sep-2023 : First Python version - Alejandro Henao
    10-Dec-2024 : Polishing            - Franklin Ayala 

    """

    tt = np.arange(1,len(burst)+1,1/fs)
    sign = np.sign(burst)
    index_cross = np.where(np.diff(sign) == 2)[0]

    Hp = []
    T = []
    for p in range(0,len(index_cross)-1):
        a = index_cross[p]
        b = index_cross[p+1]
        Hp.append(np.max(burst[a:b+1])-np.min(burst[a:b+1]))
        T.append(tt[b]-tt[a])
    Hp = np.array(Hp)
    T = np.array(T)

    # Determine the wavenumber based on the dispersion relation
    L=np.array([wave_props.wavelength(t,h) for t in T])
    k = 2*np.pi/L

    # Transference factor Kp
    Kp=np.cosh(k*zp)/np.cosh(k*h)
    Kpmin=(np.cosh(np.pi/(h-zp)*zp))/(np.cosh(np.pi/(h-zp)*h))
    for i in range(0,len(Kp)):
        if (Kp[i]<Kpmin):
            Kp[i]=Kpmin

    H = Hp/(Kp)
    H0 = np.sort(H)[::-1]
    H13 = np.nanmean(H[:int(len(H0)/3)])
    Hmx = H0[0]

    Tm = np.nanmean(T)
    Lm = np.nanmean(L)
    return (H13,Tm,Lm,Hmx)