Case Runner
===========

.. toctree::
   :maxdepth: 4

The ``run_case`` module provides functionality to finalise the XBeach ``params.txt``
configuration file. After all forcing and bathymetry inputs have been written,
:class:`CaseRunner` fills the remaining sections: the output file path, optional
gauge point locations, global and point-based output variable lists, and the
computation timing block (start time, stop time, output intervals).

.. autoclass:: oceanicospy.models.xbeachpy.execution.CaseRunner
   :members:
   :undoc-members:
   :noindex:
