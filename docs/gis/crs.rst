Coordinate reference systems
============================

.. toctree::
   :maxdepth: 2

The ``crs`` module provides tools to reproject point data between
coordinate reference systems.  Both vector files and XYZ plain-text files
are supported.  All EPSG codes can be passed as integers (e.g. ``9377``)
or strings (e.g. ``"EPSG:9377"``).

PointFileReprojector
--------------------

.. autoclass:: oceanicospy.gis.PointFileReprojector
   :members:
   :undoc-members:

reproject_points
----------------

.. autofunction:: oceanicospy.gis.reproject_points
