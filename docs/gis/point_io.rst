Point file I/O
==============

.. toctree::
   :maxdepth: 2

``PointFileIO`` reads and writes XYZ plain-text files and point vector
files (e.g. shapefiles).  The on-disk layout is described by an
``XYZFormatSpec`` dataclass, which can be provided explicitly or inferred
automatically from the file content.

XYZFormatSpec
-------------

.. autoclass:: oceanicospy.gis.XYZFormatSpec
   :members:
   :exclude-members: __init__

PointFileIO
-----------

.. autoclass:: oceanicospy.gis.PointFileIO
   :members:
   :undoc-members:
