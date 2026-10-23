Unstructured grids (DMPlex)
===========================

:cpp:class:`declib::UnstructuredGrid` wraps PETSc ``DMPlex``. Meshes are read
with ``DMPlexCreateFromFile`` (Gmsh, Exodus II, HDF5, ...), and the boundary
operators are taken from the ``DMPlex`` cone/support structure.

.. code-block:: python

   grid = declib.UnstructuredGrid("bunny.msh")

.. todo:: Document dual-mesh construction and well-centredness requirements.
