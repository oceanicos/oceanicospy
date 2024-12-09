import numpy as np
from ..utils import wave_props

def zero_crossing(burst,fs,h,zp):
    """
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

    """

    tt = np.arange(1,len(burst)+1,1/fs)
    sign = np.sign(burst)
    index_cross = np.where(np.diff(sign) == 2)[0]

    Hw = []
    T = []
    for p in range(0,len(index_cross)-1):
        a = index_cross[p]
        b = index_cross[p+1]
        Hw.append(np.max(burst[a:b])-np.min(burst[a:b]))
        T.append(tt[b]-tt[a])
    Hw = np.array(Hw)
    T = np.array(T)

    # Determinar el numero de la ola (k) con la teoría lieneal
    L = []
    for i in range(0,len(T)):
        L.append(wave_props.wavelength(T[i],h))
    L = np.array(L)
    k = 2*np.pi/L

    # Factor de transferencia Kp
    Kp=np.cosh(k*(zp))/np.cosh(k*(h))
    Kpmin=(np.cosh(np.pi/(h-zp)*zp))/(np.cosh(np.pi/(h-zp)*h))
    for i in range(0,len(Kp)):
        if (Kp[i]<Kpmin):
            Kp[i]=Kpmin

    H=Hw/(Kp)
    H = np.sort(H)
    H = H[::-1]
    H3 = H[:int(len(H)/3)]
    Hs = np.nanmean(H3)
    Tm = np.nanmean(T)
    return (Hs,Tm)