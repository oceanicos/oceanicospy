HOBO
====

.. toctree::
   :maxdepth: 3

HOBO data loggers are compact, battery-powered instruments used to measure environmental parameters such as
temperature and conductivity. The ``HOBOBase`` class provides a common interface for all HOBO logger implementations,
while ``HOBO_Temp`` and ``HOBO_TempCond`` handle temperature and conductivity loggers respectively.

.. note::
   ``HOBOBase`` is not designed to be called directly in the code but is intended to be subclassed by specific HOBO logger implementations.

.. autoclass:: oceanicospy.observations.hobo.HOBOBase
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: oceanicospy.observations.hobo.HOBO_Temp
   :members:

.. autoclass:: oceanicospy.observations.hobo.HOBO_TempCond
   :members: