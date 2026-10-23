DecLib++
========

**DecLib++** is a C++ library for *discrete exterior calculus* (DEC), with a
Python interface. It builds discrete differential forms and the operators that
act on them (exterior derivative :math:`d`, Hodge star :math:`\star`, wedge
and interior products, codifferential :math:`\delta` and Hodge Laplacian
:math:`\Delta`) for straight and twisted, scalar and vector-bundle valued
forms, on top of:

* `PETSc <https://petsc.org>`_ for parallel meshes and solvers, using
  ``DMStag`` for structured grids and ``DMPlex`` for unstructured ones;
* `Kokkos <https://kokkos.org>`_ for performance portability across CPUs and
  GPUs.

.. code-block:: python

   import declib as dl

   topo = dl.StagTopology((128, 128))
   M = dl.DiscreteManifold(topo, dl.UniformGeometry(topo))
   V0 = M.space(0, dl.Orientation.STRAIGHT)
   lap = dl.hodge_laplacian(V0)

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

   developer/architecture
   developer/contributing
   developer/writing-docs
   developer/testing

Indices
-------

* :ref:`genindex`
* :ref:`search`
