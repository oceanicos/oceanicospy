Initializer
===========

.. toctree::
   :maxdepth: 3

The Initializer module is responsible for setting up the XBeach project structure.
It creates the required directory tree (``input/``, ``pros/``, ``run/``, ``output/``),
copies the bundled ``params_base.txt`` template into ``run/``, and stamps it with the
user-supplied configuration flags.

.. autoclass:: oceanicospy.models.xbeachpy.Initializer
   :members:
