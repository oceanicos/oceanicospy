BaseLogger
==========

Because all the pressure sensors share a common data structure, which basically consists of a time series of pressure values, 
the ``BaseLogger`` class is designed to provide a common interface for all those sensors. 
This class serves as a parent (base) class for specific logger implementations, such as AQUAlogger, RBR, and BlueLog.

.. note::
   This class is not designed to be called directly in the code but is intended to be subclassed by specific logger implementations.

.. automodule:: oceanicospy.observations.pressure_sensor_base
   :members:
   :undoc-members:
   :show-inheritance: