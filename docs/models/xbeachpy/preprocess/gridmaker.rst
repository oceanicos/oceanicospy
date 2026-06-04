GridMaker
=========

.. toctree::
   :maxdepth: 4

The ``gridmaker`` module is responsible for generating the XBeach computational grid.
It follows a builder pattern: the main :class:`GridMaker` class exposes two properties
that return lightweight builder objects — one for 1-D cross-shore profile grids and one
for 2-D rectangular grids — each delegating coordinate geometry to the corresponding
GIS data class (:class:`~oceanicospy.gis.ProfileAxis` or :class:`~oceanicospy.gis.Grid`).

GridMaker
---------

.. autoclass:: oceanicospy.models.xbeachpy.preprocess.GridMaker
   :members:
   :undoc-members:
   :noindex:

Profile builder
---------------

.. autoclass:: oceanicospy.models.xbeachpy.preprocess.gridmaker._ProfileBuilder
   :members:
   :undoc-members:
   :noindex:

Rectangular grid builder
------------------------

.. autoclass:: oceanicospy.models.xbeachpy.preprocess.gridmaker._RectangularGridBuilder
   :members:
   :undoc-members:
   :noindex:
