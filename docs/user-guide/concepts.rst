Core concepts
=============

DecLib++ separates a discretization into a few independent pieces. Each
piece is a class in C++ and in Python.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Concept
     - C++ / Python
     - Role
   * - Topology
     - :cpp:class:`declib::Topology` / :py:class:`declib.topology.Topology`
     - Oriented cell complex, its dual, incidence and parallel layout
   * - Grid mapping
     - :cpp:class:`declib::GridMapping` / :py:class:`declib.mapping.GridMapping`
     - Which complex hosts straight forms and which hosts twisted forms
   * - Geometry
     - :cpp:class:`declib::Geometry` / :py:class:`declib.geometry.Geometry`
     - Coordinates, dual vertex placement, cell measures
   * - Discrete manifold
     - :cpp:class:`declib::DiscreteManifold` / :py:class:`declib.manifold.DiscreteManifold`
     - The three above combined; factory for form spaces
   * - Forms
     - :cpp:class:`declib::FormSpace`, :cpp:class:`declib::Form`
     - Discrete differential forms, scalar or vector-bundle valued
   * - Operators
     - :cpp:class:`declib::LinearOperator`, :cpp:class:`declib::BilinearOperator`,
       :cpp:class:`declib::NonlinearOperator`, :cpp:class:`declib::Functional`
     - :math:`d`, :math:`\star`, :math:`\wedge`, :math:`i_X`, :math:`i_X^\dagger` and compositions

Topology and the dual complex
-----------------------------

A topology is a cell complex of dimension :math:`n`: vertices (:math:`k=0`),
edges, faces and volumes, each with an inner orientation. Every primal
:math:`k`-cell is crossed by one dual :math:`(n-k)`-cell, which carries the
outer orientation induced by it. Dual cells are always indexed by the primal
cell they cross.

On a non-periodic grid, :cpp:enum:`declib::DualBoundary` says how the dual
complex is closed. The default, ``Extended``, adds dual cells on the boundary
itself, so the discrete Stokes theorem holds on the dual complex including
boundary terms.

Straight and twisted forms
--------------------------

A straight (inner oriented) :math:`k`-form is integrated over cells with an
inner orientation; a twisted (outer oriented) :math:`k`-form over cells with
an outer orientation. The :cpp:class:`declib::GridMapping` decides where each
lives:

.. list-table::
   :header-rows: 1

   * - Form (:math:`n = 2`)
     - ``straight_on_primal``
     - ``twisted_on_primal``
   * - straight 0-form
     - primal vertices
     - dual vertices
   * - straight 1-form
     - primal edges
     - dual edges
   * - twisted 1-form
     - dual edges
     - primal edges
   * - twisted 2-form
     - dual cells
     - primal cells

Under either mapping a straight :math:`k`-form and a twisted
:math:`(n-k)`-form sit on the same primal cells, so a diagonal Hodge star is a
pointwise scaling.

Forms
-----

A :cpp:struct:`declib::FormType` is a degree, an orientation and a
:cpp:class:`declib::Bundle` (the fiber). Form types are runtime values:
operators check them once, when they are built.

A :math:`k`-form is stored as a cochain, one fiber value per cell, through the
de Rham map

.. math::

   R(\omega)_\sigma = \int_\sigma \omega .

Vector-bundle valued forms store :math:`r` values per cell. For trivial
bundles operators act componentwise; other bundles go through a
:cpp:class:`declib::Connection`.

Operators
---------

Exterior derivative
   :math:`d_k = \partial_{k+1}^{T}` on the primal complex, so the discrete
   Stokes theorem is exact and :math:`d \circ d = 0`. Metric-free. See
   :cpp:class:`declib::ExteriorDerivative`.

Duality pairing
   :math:`\langle \alpha, \beta \rangle`, the discrete
   :math:`\int_M \alpha \wedge \beta` between a :math:`k`-form and a form of
   the complementary type. Metric-free. See :cpp:class:`declib::Pairing`.

Hodge star
   Maps :math:`k`-forms to :math:`(n-k)`-forms of the opposite orientation and
   carries the metric. See :cpp:class:`declib::HodgeStar`.

Wedge and interior product
   Bilinear operators. The interior product's pairing adjoint
   :math:`i_X^\dagger` satisfies
   :math:`\langle i_X \alpha, \beta \rangle = \langle \alpha, i_X^\dagger \beta \rangle`
   exactly. See :cpp:class:`declib::Wedge`, :cpp:class:`declib::InteriorProduct`.
   Concrete schemes for both are still to be chosen.

Derived operators
   :math:`\delta = (-1)^{n(k+1)+1} \star d \star`,
   :math:`\Delta = d\delta + \delta d`, the mass matrix and the Lie derivative
   :math:`L_X = d\, i_X + i_X\, d`. See :cpp:func:`declib::hodge_laplacian`.

Boundary conditions
-------------------

Essential conditions are constrained degrees of freedom on a form space
(:cpp:struct:`declib::EssentialCondition`). Natural conditions are values on
the extended dual boundary cells (:cpp:func:`declib::Form::set_boundary_values`).
