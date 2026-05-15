Theoretical Foundations
=======================

.. toctree::
   :maxdepth: 4

Introduction
------------

Ocean tides are among the most predictable geophysical phenomena on Earth.
They arise from the periodic gravitational forcing exerted by the Moon and
the Sun on the rotating Earth, modulated by the geometry of ocean basins and
local coastal morphology. Despite this astronomical regularity, the tidal
signal recorded at any given coastal station is a complex superposition of
oscillations occurring at dozens of distinct frequencies, a structure that
harmonic analysis is specifically designed to disentangle.

**Harmonic tidal decomposition** is the standard technique for separating the
tidal signal from a sea level time series. The core idea is to represent the
observed sea surface height :math:`\eta(t)` as a finite sum of sinusoidal
oscillations, each associated with a known astronomical frequency. Once the
amplitude and phase of each oscillation are determined by fitting the model to
observations, the tidal signal can be reconstructed at any point in time,
and, by subtraction, the **non-tidal residual** (capturing storm surges,
seiches, long-period climate variability, and instrumental noise) can be
isolated.

``tide_harmonic_decomposition`` performs this analysis using
`UTide <https://github.com/wesleybowman/UTide>`_, a Python implementation of
the unified tidal analysis framework described by Codiga (2011), which is
itself grounded in decades of classical tidal theory.


.. _tidal_constituents:

Tidal Constituents and the Harmonic Model
------------------------------------------

The mathematical foundation of tidal harmonic analysis is the representation
of sea level as a superposition of sinusoids:

.. math::

   \eta(t) = Z_0 + \sum_{k=1}^{N} A_k \cos\!\left(\omega_k t - \phi_k\right) + \varepsilon(t)

where:

- :math:`Z_0` is the mean sea level (datum offset),
- :math:`N` is the number of tidal constituents included in the fit,
- :math:`A_k` is the amplitude of the :math:`k`-th constituent,
- :math:`\omega_k` is its angular frequency (radians per hour), determined
  by astronomical theory,
- :math:`\phi_k` is its phase lag relative to the equilibrium tide
  (expressed in degrees, referred to Greenwich),
- :math:`\varepsilon(t)` is the non-tidal residual.

Each constituent is identified by a standard name, for example, **M2**
(the principal lunar semidiurnal tide), **S2** (the principal solar
semidiurnal tide), **K1** and **O1** (the major diurnal tides). Their
frequencies are not free parameters; they are prescribed precisely by the
orbital mechanics of the Earth–Moon–Sun system. Only the amplitudes
:math:`A_k` and phases :math:`\phi_k` are estimated from data, making
harmonic analysis a well-constrained regression problem.

.. note::

   The latitude parameter ``lat`` passed to ``tide_harmonic_decomposition``
   is used by UTide to compute the **nodal corrections** — small,
   time-varying adjustments to the amplitude and phase of each constituent
   that arise from the 18.6-year precession of the Moon's orbital plane. These
   corrections are essential for accurate tidal prediction, particularly over
   long records.

The harmonic representation can equivalently be written in the linear
(cosine–sine) form convenient for regression:

.. math::

   \eta(t) = Z_0 + \sum_{k=1}^{N}
   \left[ u_k \cos(\omega_k t) + v_k \sin(\omega_k t) \right]
   + \varepsilon(t)

where :math:`u_k = A_k \cos\phi_k` and :math:`v_k = A_k \sin\phi_k` are
the in-phase and quadrature coefficients. This is the form actually solved
during regression; amplitude and phase are recovered afterwards as
:math:`A_k = \sqrt{u_k^2 + v_k^2}` and
:math:`\phi_k = \arctan(v_k / u_k)`.


.. _fitting_methods:

Regression Methods (``method`` parameter)
------------------------------------------

The harmonic coefficients :math:`(u_k, v_k)` for all constituents are
estimated simultaneously by solving the linear regression system
:math:`\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}`,
where :math:`\mathbf{y}` is the vector of observed sea level values and
:math:`\mathbf{X}` is the design matrix containing the evaluated cosine and
sine basis functions at each observation time. Two fitting strategies are
supported:

**Ordinary Least Squares** (``method='ols'``)
   OLS minimises the sum of squared residuals:

   .. math::

      \hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}}
      \left\| \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right\|^2
      = \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}

   This estimator is optimal (minimum variance, unbiased) when the residuals
   are independent and identically distributed (i.i.d.) with zero mean.
   It is the default and is well suited to records with low to moderate
   non-tidal variability.

**Iteratively Reweighted Least Squares — Robust** (``method='robust'``)
   In the presence of outliers, large storm-surge events, or strongly
   non-Gaussian residuals, OLS can yield biased constituent estimates because
   extreme values carry disproportionate weight in the squared-error
   objective. The robust method applies iteratively reweighted least squares
   (IRLS) with a bi-square (Tukey) weight function:

   .. math::

      \hat{\boldsymbol{\beta}}^{(r+1)} =
      \left(\mathbf{X}^\top \mathbf{W}^{(r)} \mathbf{X}\right)^{-1}
      \mathbf{X}^\top \mathbf{W}^{(r)} \mathbf{y}

   where :math:`\mathbf{W}^{(r)}` is a diagonal matrix of observation weights
   that downweight large residuals at iteration :math:`r`. Convergence
   typically requires five to ten iterations. The robust estimator is
   recommended when the time series contains episodic, high-amplitude
   non-tidal signals that would otherwise contaminate the harmonic fit.


.. _confidence_intervals:

Confidence Interval Estimation (``conf_int`` parameter)
---------------------------------------------------------

Uncertainty quantification is a critical component of tidal analysis because
constituent amplitudes and phases derived from finite, noisy records are
themselves random variables. ``tide_harmonic_decomposition`` supports three
approaches to estimating the 95% confidence intervals reported in
``coefTable`` (columns ``A_ci`` and ``P_ci``):

**Monte Carlo simulation** (``conf_int='MC'``)
   The residual time series :math:`\hat{\varepsilon}(t)` is used to
   characterise the background noise. A large ensemble of surrogate noise
   realisations (preserving the spectral structure of the residual) is
   generated and added back to the reconstructed tide. Each realisation is
   re-analysed harmonically, yielding an empirical distribution of fitted
   amplitudes and phases. The 2.5th and 97.5th percentiles of these
   distributions define the 95% confidence intervals. Monte Carlo intervals
   are non-parametric and make no assumptions about the distribution of
   errors, making them the most reliable choice — especially for short
   records or coloured noise.

**Linearised (analytical) approximation** (``conf_int='linearized'``)
   Confidence intervals are derived analytically from the covariance matrix
   of the OLS estimator,
   :math:`\mathbf{C}_{\hat{\beta}} = \hat{\sigma}^2
   (\mathbf{X}^\top\mathbf{X})^{-1}`, propagated to amplitude and phase
   via the delta method. This approach is computationally inexpensive but
   assumes that residuals are white noise (uncorrelated and Gaussian).
   In practice, tidal residuals often exhibit some spectral colouring, so
   linearised intervals may be overly optimistic — i.e., narrower than the
   true uncertainty.

**No confidence intervals** (``conf_int='none'``)
   Uncertainty estimation is skipped entirely. This reduces computation
   time and is acceptable during exploratory analysis or when processing
   large numbers of stations, provided that the resulting uncertainties are
   not required for downstream interpretation.

.. tip::

   For scientific reporting or comparison of constituent amplitudes across
   stations, ``conf_int='MC'`` is strongly recommended. The additional
   computational cost is modest for typical coastal time series of months
   to years in length.


.. _pe_snr:

Goodness-of-Fit Statistics: PE and SNR
----------------------------------------

Two diagnostic quantities are computed for each constituent and stored in
``coefTable``. They serve as the primary criteria for judging whether a
constituent is reliably resolved by the available record.

Percentage of Energy (``PE``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Percentage of Energy** quantifies the relative contribution of each
tidal constituent to the total tidal variance. For the :math:`k`-th
constituent it is defined as:

.. math::

   \mathrm{PE}_k = \frac{A_k^2 / 2}{\sum_{j=1}^{N} A_j^2 / 2} \times 100\%
   = \frac{A_k^2}{\sum_{j=1}^{N} A_j^2} \times 100\%

where the factor :math:`1/2` cancels and the result is simply the ratio of
individual to total constituent variance, expressed as a percentage.

PE provides a physically intuitive ranking of tidal constituents:

- In a **semidiurnal-dominated** tidal regime (common along Atlantic coasts),
  M2 typically accounts for 60–80% of tidal energy, with S2 contributing
  10–20%.
- In a **diurnal-dominated** regime (e.g., the Gulf of Mexico), K1 and O1
  together can carry more than 50% of total tidal energy.
- Constituents with very small PE values (e.g., < 0.1%) are energetically
  negligible and their precise estimation requires very long records.

Signal-to-Noise Ratio (``SNR``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Signal-to-Noise Ratio** measures how well a constituent stands above the
background noise level at its frequency:

.. math::

   \mathrm{SNR}_k = \frac{A_k^2}{\sigma_k^2}

where :math:`\sigma_k^2` is the estimated noise variance in the frequency
band surrounding constituent :math:`k`. A constituent is considered
**statistically significant** when:

.. math::

   \mathrm{SNR}_k > 2

This threshold corresponds approximately to a signal amplitude exceeding
:math:`\sqrt{2}` times the local noise standard deviation, and is the
conventional acceptance criterion in tidal analysis (Pawlowicz et al., 2002;
Codiga, 2011).

Practical interpretation:

- **SNR > 10**: The constituent is very well resolved; its amplitude and phase
  estimates are highly reliable.
- **2 < SNR ≤ 10**: The constituent is marginally to moderately significant.
  It should be retained in the tidal prediction but its uncertainty is
  non-trivial.
- **SNR ≤ 2**: The constituent cannot be distinguished from noise at the
  available record length or sampling rate. Amplitudes and phases for these
  constituents are unreliable and should not be used for tidal prediction or
  physical interpretation.

.. warning::

   Constituents with SNR ≤ 2 are returned in ``coefTable`` but flagged
   implicitly by their low SNR. It is strongly recommended to filter
   significant constituents before using results in prediction or further
   analysis::

      significant = coefTable[coefTable['SNR'] > 2]


.. _residual_interpretation:

The Non-Tidal Residual
-----------------------

After reconstruction of the tidal signal, the **residual**
:math:`r(t) = \eta(t) - \hat{\eta}_{\text{tide}}(t)` represents all
variability in sea level that is not explained by the fitted harmonic
constituents. This signal is of independent oceanographic interest and may
contain contributions from:

- **Meteorological storm surges**: wind stress and inverse barometer effects
  driven by atmospheric pressure anomalies;
- **Seiches**: resonant free oscillations of semi-enclosed basins at periods
  from minutes to hours;
- **Sub-tidal and mesoscale variability**: boundary currents, eddies, and
  coastal-trapped waves;
- **Long-term sea level trends**: associated with steric changes, freshwater
  input, or vertical land motion.

The quality of the residual depends directly on the fidelity of the harmonic
fit. Using a sufficient number of significant constituents (SNR > 2) and
choosing an appropriate regression method minimises the leakage of tidal
energy into the residual.


References
----------

Codiga, D. L. (2011). *Unified tidal analysis and prediction using the UTide
Matlab functions*. Graduate School of Oceanography, University of Rhode Island.
Technical Report 2011-01.

Foreman, M. G. G. (1977). *Manual for Tidal Heights Analysis and Prediction*.
Pacific Marine Science Report 77-10. Institute of Ocean Sciences, Patricia Bay.

Pawlowicz, R., Beardsley, B., & Lentz, S. (2002). Classical tidal harmonic
analysis including error estimates in MATLAB using T_TIDE. *Computers &
Geosciences*, 28(8), 929–937. https://doi.org/10.1016/S0098-3004(02)00013-4