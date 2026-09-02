Wind Forcing
============

.. toctree::
   :maxdepth: 4

The ``wind_forcing`` module preprocesses wind boundary conditions for XBeach simulations.
It integrates with the **ERA5** reanalysis (via the CDS API) and the **CMDS** dataset
to download hourly 10-m wind components, extracts a representative single-point time
series at a user-specified location, converts it to nautical speed and direction, and
writes the two-column ASCII file expected by XBeach.

.. autoclass:: oceanicospy.models.xbeachpy.preprocess.WindForcing
   :members:
   :undoc-members:
   :noindex:
