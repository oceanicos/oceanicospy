XYZ masking
===========

.. toctree::
   :maxdepth: 2

``XYZRectangleMask`` filters an XYZ dataset by one or more axis-aligned
rectangular regions.  Two modes are available: ``"keep"`` retains only
points inside the rectangles and ``"exclude"`` removes them.  Rectangles
are defined by an ``AxisAlignedRectangle`` using two opposite corner
coordinates in the same CRS as the data.

AxisAlignedRectangle
--------------------

.. autoclass:: oceanicospy.gis.AxisAlignedRectangle
   :members:
   :undoc-members:

XYZRectangleMask
----------------

.. autoclass:: oceanicospy.gis.XYZRectangleMask
   :members:
   :undoc-members:
