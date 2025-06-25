Signal conditioning
=================

Most of the time series recorded by different devices need to be conditioned before
using them in spectral analysis. Different key concepts will be explained in this section
to obtain the most feasible results.

Nyquist frequency
-----------------

It is defined as the minimum sampling rate that allows to reconstruct a signal. This frequency is given by the equation:

.. math::
    f_{Nyquist} = \frac{f_s}{2}

where :math:`f_s` is the sampling frequency.

For instance, if you want to record a signal with a minimum frequency of 0.5 Hz (2s period), 
you need to sample it at least at 1 Hz. If a lower frequency signal is required, the sampling rate 
must be increased. At the end, there is a linear relationship between the sampling frequency and the
maximum frequency that can be processed. Usually a sampling rate of 1Hz is enough for wind-wave analysis.

.. figure:: ../images/nyquist_example.png
    :alt: Example of Nyquist frequency
    :align: center
    :width: 70%

    Illustration of the Nyquist frequency concept. The sampling rate determines the highest frequency that can be accurately represented.

Aliasing
--------




