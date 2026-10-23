Structured grids (DMStag)
=========================

:cpp:class:`declib::stag::Topology` builds a logically Cartesian complex in
1, 2 or 3 dimensions on PETSc ``DMStag``. A staggered grid stores values on
vertices, edges, faces and elements, which are exactly the strata of the
cubical complex. The dual of a cubical complex is cubical too, and every dual
cell is stored on the stratum of the primal cell it crosses, so one ``DMStag``
decomposition serves every form type.

.. code-block:: cpp

   using namespace declib;
   auto topo = stag::Topology<3>::create(PETSC_COMM_WORLD,
       {.cells = {64, 64, 64},
        .boundary = {stag::Boundary::Periodic, stag::Boundary::Periodic, stag::Boundary::Ghosted}});
   auto geom = stag::UniformGeometry<3>::create(topo, {.lengths = {1.0, 1.0, 0.5}});

Storage
-------

.. list-table::
   :header-rows: 1

   * - Primal cell (3D)
     - ``DMStag`` location
     - Dual cell stored there
   * - vertex
     - ``BACK_DOWN_LEFT``
     - dual 3-cell
   * - x-, y-, z-edge
     - ``BACK_DOWN``, ``BACK_LEFT``, ``DOWN_LEFT``
     - dual 2-cell
   * - yz-, zx-, xy-face
     - ``LEFT``, ``DOWN``, ``BACK``
     - dual 1-cell
   * - element
     - ``ELEMENT``
     - dual 0-cell

Each :cpp:class:`declib::stag::FormSpace` has its own compatible ``DMStag``
(``DMStagCreateCompatibleDMStag``) with dofs only on its strata, and vectors of
type ``VECKOKKOS``. Kernels read values through
:cpp:class:`declib::stag::FormView`.

Orientation conventions
-----------------------

Primal edges point along the positive axis, faces are oriented by their normal
through the right-hand rule, and elements are positively oriented. Dual cells
carry the outer orientation induced by the primal cell they cross. The sign
tables of every operator are generated from ``declib/stag/orientation.hpp``.

Geometries
----------

:cpp:class:`declib::stag::UniformGeometry` is available now. Rectilinear
(tensor-product spacing) and mapped (curvilinear) geometries follow with the
same interface.
