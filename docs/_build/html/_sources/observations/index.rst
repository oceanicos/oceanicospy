Observations
=============

This subpackage contains classes for handling observations from various sensors. Each class is designed to interface 
with a specific type of logger or sensor, providing methods for practical data retrieval. The sensors described here represent the whole existing set
of sensors in the OCEANICOS research group used in 2025.

The sensors include:

- **Pressure sensors**: devices used to measure the water pressure at a given depth. Some of these sensors include an own-developed logger (BlueLog) and
  comercial loggers such as RBR and AQUAlogger.

- **Acoustic doppler current profilers (ADCP)**: instruments that use sound waves to measure water current velocities.
  The AWAC is an example of such a device.

- **Weather stations**: Despite the OCEANICOS research group is not an atmospheric research group, it is unneglectable the need of weather data for
  oceanographic research. In this group the examples of those stations are: ..

- **Resistive sensors**: The lab sensors are used to measure the resistivity of the water, which can be used to infer properties like water depth.

- **Conductivity-Temperature-Depth (CTD) sensors**: These sensors measure the conductivity, temperature, and depth of water, providing essential data for
  oceanographic studies.

.. note::
   This documentation is generated from the source code and may not include all details.
   For more information, refer to the source code or the specific sensor documentation.

All the classes in this subpackage are presented below and it can be imported as follows:


.. hint::
   To use a specific sensor, you can import it directly from the `oceanicospy.observations` module.

   .. code-block:: python

      from oceanicospy.observations import device_name_class
      device = device_name_class()  # name of the specific sensor class you want to use.

   For example, to use the AQUAlogger:

   .. code-block:: python

      from oceanicospy.observations import AQUAlogger
      aqua_logger = AQUAlogger()  # Create an instance of the AQUAlogger class.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   walkthrough
   BaseLogger
   AQUAlogger
   RBR
   AWAC