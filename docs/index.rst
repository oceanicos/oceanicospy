oceanicospy documentation
=========================

**oceanicospy** is a Python library for handling in-situ oceanographic data analysis,
reading observations from a diverse set of sensors, automating numerical modelling
workflows, downloading data from common sources, and working with GIS data.

In the following diagram, the high-level structure of the library is shown, **oceanicospy** is organized into subpackages
that group related functionality together.

.. code-block:: text

   oceanicospy/
   ├── analysis/      temporal and spectral analysis
   ├── gis/           geospatial utilities for coastal data
   ├── models/        preprocessing for numerical models (SWAN, XBeach)
   ├── observations/  instrument readers
   ├── plots/         visualization utilities
   ├── retrievals/    data retrieval from ERA5, CMEMS, UHSLC
   └── utils/         shared helpers used across subpackages

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installing
   dev_setup

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   analysis/index
   observations/index
   downloads/index
   models/index
   gis/index

.. toctree::
   :maxdepth: 1
   :caption: Development & Support

   contributing
   support
