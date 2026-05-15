Weather Station Base
====================

.. toctree::
   :maxdepth: 3

Because all weather stations share a common data structure consisting of atmospheric measurements such as wind speed,
temperature, and precipitation, the ``WeatherStationBase`` class is designed to provide a common interface for all those stations.
This class serves as a parent (base) class for specific weather station implementations, such as DavisVantagePro, WeatherSens, and Rainwise.

.. note::
   This class is not designed to be called directly in the code but is intended to be subclassed by specific weather station implementations.

.. automodule:: oceanicospy.observations.weather_stations.weather_station_base
   :members:
   :undoc-members:
   :show-inheritance:
