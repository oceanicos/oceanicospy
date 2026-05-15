Profile interpolator
====================

.. toctree::
   :maxdepth: 2

``ProfileInterpolator`` samples Z values from a scattered XYZ point cloud
at the positions defined by a :class:`~oceanicospy.gis.ProfileAxis`.  The
pipeline pre-filters the point cloud to a corridor around the profile line,
builds a k-d tree for fast nearest-neighbour lookup, and returns the
interpolated profile as a ``pandas.DataFrame``.

The axis must be built with
:meth:`~oceanicospy.gis.ProfileAxis.from_coordinates` since planar
coordinates are required to locate the profile within the point cloud.

.. autoclass:: oceanicospy.gis.ProfileInterpolator
   :members:
   :undoc-members:
