Water Level Forcing
===================

.. toctree::
   :maxdepth: 4

The ``waterlevel_forcing`` module preprocesses water-level boundary conditions for
XBeach simulations. It connects to the **University of Hawaii Sea Level Center (UHSLC)**
research-quality gauge archive to download hourly sea-level records, converts them to the
two-column (elapsed seconds, sea level) ASCII format expected by XBeach, and writes the
relevant section of ``params.txt``.

Additional methods support data from the **CECOLDO** tide-gauge network and allow
deployment of pre-existing water-level files.

.. autoclass:: oceanicospy.models.xbeachpy.preprocess.WaterLevelForcing
   :members:
   :undoc-members:
   :noindex:
