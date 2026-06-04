Observations
=============

This subpackage contains classes for handling observations from various sensors. Each class is designed to interface 
with a specific type of logger/sensor, providing methods for practical data retrieval. The sensors described here represent an existing set
of sensors in OCEANICOS research group, however, the package is designed to be easily extendable to include new sensors in the future. 
The main goal of this subpackage is to provide a unified interface for accessing and processing data from different types of sensors, 
making it easier for researchers to work with their observational data.

The sensors include:

- **Pressure sensors**: devices used to measure the water pressure at a given depth. Some of these sensors include an own-developed logger (Bluelog) and
  comercial loggers such as RBR and AQUAlogger.

- **Weather stations**: Despite the OCEANICOS research group is not an atmospheric research group, it is unneglectable the need of weather data for
  oceanographic research. In this group the examples of those stations are: Davis Vantage Pro, Rainwise and WeatherSens.

- **Conductivity-Temperature-Depth (CTD) sensors**: These sensors measure the conductivity, temperature, and depth of water, providing essential data for
  oceanographic studies. The current supported models are: CTD 48M from Sea Sun Marine Tech and Castaway-CTD from Xylem.

- **Acoustic doppler current profilers (ADCP)**: this instrument use sound waves to measure water current velocities. The AWAC from Nortek is the only supported model for now.

- **Buoy**: this class is designed to handle data from buoys. The Spotter model from SOFAR is currently supported.

- **HOBO**: this class is designed to handle data from HOBO loggers, which are commonly used for environmental monitoring.

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

      from oceanicospy.observations import pressure_sensors
      aqua_logger = pressure_sensors.AQUAlogger()  # Create an instance of the AQUAlogger class.

   You can easily check the available classes and their methods in the documentation or by exploring the source code.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   pressure_sensors/index
   weather_stations/index
   ctd/index
   AWAC
   Buoy
   HOBO

.. toctree::
   :maxdepth: 1
   :caption: Example:
   
   reading_field_data
