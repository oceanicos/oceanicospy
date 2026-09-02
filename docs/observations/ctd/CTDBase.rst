CTDBase
=======

.. toctree::
   :maxdepth: 3

Because all CTD sensors share a common data structure consisting of conductivity, temperature, and depth time series,
the ``CTDBase`` class is designed to provide a common interface for all those sensors.
This class serves as a parent (base) class for specific CTD implementations, such as CastawayCTD and CTD48M.

.. note::
   This class is not designed to be called directly in the code but is intended to be subclassed by specific CTD implementations.

.. autoclass:: oceanicospy.observations.ctd.ctd_base.CTDBase
   :members:
   :undoc-members:
   :show-inheritance:
