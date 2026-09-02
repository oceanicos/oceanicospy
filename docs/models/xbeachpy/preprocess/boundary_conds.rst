Boundary Conditions
===================

.. toctree::
   :maxdepth: 4

The ``boundary_conds`` module generates XBeach spectral boundary condition files from
SWAN output. Starting from a ``SpecSWAN.out`` file it converts the wave spectra to
XBeach-compatible ``.sp2`` files, writes the corresponding ``filelist_<n>.txt``
filelists and ``loclist.txt``, and populates the boundary block of ``params.txt``.

.. autoclass:: oceanicospy.models.xbeachpy.preprocess.BoundaryConditions
   :members:
   :undoc-members:
   :noindex:
