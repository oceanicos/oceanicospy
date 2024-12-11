import numpy as np
from . import constants

def wavelength(T,h):
    """
    Computes the wavelength for intermediate waters
    
    Parameters
    ----------
    T: float
        wave period
    h: float
        depth      
    
    Returns
    -------
    L : wavelength

    Notes
    -----
    01-Sep-2023 : First Python function - Juan Diego Toro

    """

    Lo = (constants.GRAVITY*T**2)/(2*np.pi);
    L1  = (constants.GRAVITY*T**2)/(2*np.pi)*np.tanh((h*2*np.pi)/Lo);
    i = 0;
    while (abs(Lo-L1)>0.0001):
        Lo = L1;
        L1  = (constants.GRAVITY*T**2)/(2*np.pi)*np.tanh(h*2*np.pi/Lo);
        i = i+1;
        if (i>5000):
            break
    return (L1)

def direction(vn,ve):
    """
    Computes the direction w.r.t north
    
    Parameters
    ----------
    vn: float
        north velocity
    ve: float
        east velocity
    
    Returns
    -------
    d_degrees: float
        wave direction in degrees w.r.t. north

    Notes
    -----
    01-Sep-2023 : First Python function - Juan Diego Toro

    """

    if vn == float(0):
        if ve < 0:
            d=-np.pi/2
        else:
            d=np.pi/2
    else:
        if vn>0:
            d=np.arctan(ve/vn)
        elif vn<0:
            if ve<0:
                d=np.arctan(ve/vn)-np.pi
            else:
                d=np.arctan(ve/vn)+np.pi
    d_degrees = d*180/np.pi
    
    return d_degrees