DecLib++
========

**DecLib++** is a C++ library for *discrete exterior calculus* (DEC), with a
Python interface. It builds discrete differential forms and the operators that
act on them (exterior derivative :math:`d`, Hodge star :math:`\star`,
codifferential :math:`\delta` and Hodge Laplacian :math:`\Delta`) on top of:

* `PETSc <https://petsc.org>`_ for parallel meshes and solvers, using
  ``DMStag`` for structured grids and ``DMPlex`` for unstructured ones;
* `Kokkos <https://kokkos.org>`_ for performance portability across CPUs and
  GPUs.

.. code-block:: python

   import declib

   grid = declib.StructuredGrid((128, 128))
   u = declib.Cochain(grid, degree=0)
   lap_u = declib.laplacian(u)

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user-guide/concepts
   user-guide/structured-grids
   user-guide/unstructured-grids
   user-guide/performance-portability

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/cpp
   api/python

.. toctree::
   :maxdepth: 1
   :caption: Developer guide

   developer/contributing
   developer/writing-docs
   developer/testing

Indices
-------

* :ref:`genindex`
* :ref:`search`
