XYZ tile merging
================

.. toctree::
   :maxdepth: 2

``XYZMerger`` combines multiple XYZ tiles into a single point dataset.
Overlapping regions are resolved by priority: points from lower-priority
tiles that fall within the coverage footprint of any higher-priority tile
are discarded.  All input tiles must share the same CRS and on-disk format
before merging.

.. autoclass:: oceanicospy.gis.XYZMerger
   :members:
   :undoc-members:
