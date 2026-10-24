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
   * - xy-, xz-, yz-face
     - ``BACK``, ``DOWN``, ``LEFT``
     - dual 1-cell
   * - element
     - ``ELEMENT``
     - dual 0-cell

Strata are numbered by dimension, then by the bit mask of the axes they span
(x = 1, y = 2, z = 4), as in :cpp:func:`declib::stag::Topology::all_strata`;
in 3D: vertex, x-, y-, z-edge, xy-, xz-, yz-face, element. Form components of
a :math:`k`-form use the same order for the basis
:math:`dx^{i_1} \wedge \dots \wedge dx^{i_k}`.

Each :cpp:class:`declib::stag::FormSpace` has its own compatible ``DMStag``
(``DMStagCreateCompatibleDMStag``) with dofs only on its strata, and vectors of
type ``VECKOKKOS``. Kernels read values through
:cpp:class:`declib::stag::FormView`.

Orientation conventions
-----------------------

A primal :math:`k`-cell spanning the axes :math:`a_1 < \dots < a_k` carries
the orientation of :math:`dx^{a_1} \wedge \dots \wedge dx^{a_k}`: edges point
along the positive axis, faces are oriented as xy, xz and yz, and elements are
positively oriented. A dual cell spans the axes complementary to the primal
cell it crosses and is oriented the same way, by its sorted axes. Twisted
forms are identified with straight forms through the standard orientation of
:math:`\mathbb{R}^n`. With these conventions

- :math:`(d\omega)_\sigma = \sum_m (-1)^m (\omega_{\sigma^+_{b_m}} - \omega_{\sigma^-_{b_m}})`
  on either complex, where :math:`\sigma^\pm_b` are the faces at the high and
  low end of the :math:`m`-th axis :math:`b_m` of :math:`\sigma`;
- the diagonal star is :math:`\star dx^A = s_A\, dx^{A^c}` times the ratio of
  measures, with :math:`s_A` the sign of :math:`dx^A \wedge dx^{A^c}`;
- the pairing of a cell and its dual cell carries the same sign :math:`s_A`,
  so :math:`\langle a, \star a \rangle > 0`.

All signs come from ``declib/stag/orientation.hpp``.

Boundaries
----------

Every axis is periodic or not. On a non-periodic axis the dual complex is
closed with the extended dual cells (``DualBoundary::Extended``): dual cells
crossing boundary-plane primal cells are half cells, and dual cells lying in
the boundary itself carry the boundary data of forms hosted on the dual
complex. Phase 1 uses homogeneous data there, so for a 0-form on the dual
complex the Laplacian imposes :math:`u = 0` on the boundary.

Forms hosted on the primal complex take essential conditions as constrained
degrees of freedom: :cpp:func:`declib::FormSpace::with_essential` records
them and :cpp:func:`declib::apply_essential` imposes them on an assembled
system with ``MatZeroRowsColumns``.

Geometries
----------

:cpp:class:`declib::stag::UniformGeometry` is available now. Rectilinear
(tensor-product spacing) and mapped (curvilinear) geometries follow with the
same interface.
