Structured grids (DMStag)
=========================

:cpp:class:`declib::StructuredGrid` wraps PETSc ``DMStag``. A staggered grid
stores unknowns on vertices, edges, faces and elements, which correspond
exactly to primal 0-, 1-, 2- and 3-cochains, so no extra incidence data is
needed.

.. code-block:: cpp

   declib::StructuredGrid<3> grid({64, 64, 64}, {1.0, 1.0, 1.0});

.. todo:: Document boundary types, periodicity and the dual-grid layout.
