GIS
===

The ``gis`` subpackage provides tools for reading, writing, reprojecting,
merging and masking XYZ point-cloud data, building spatial grids and
sampling bathymetric profiles.  It covers the full preprocessing pipeline
from raw survey files to analysis-ready inputs for numerical models.

.. toctree::
   :maxdepth: 1
   :caption: API reference:

   point_io
   crs
   xyz_merger
   xyz_mask
   profile_axis
   profile_interpolator
   grid

.. toctree::
   :maxdepth: 1
   :caption: Example:

   gis_examples
