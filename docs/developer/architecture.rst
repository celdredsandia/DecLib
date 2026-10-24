Architecture
============

This page records the class design and the decisions behind it. The headers
under ``include/declib`` are the authoritative interface; this page explains
how they fit together.

The full `design document <../declibpp-design.html>`_ that preceded the
implementation is kept in the repository as ``docs/design/declibpp-design.html``.
It gives the reasoning behind each class in more depth, with notes on where
phase 1 departed from it.

Layers
------

Each layer depends only on the layers below it.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Layer
     - Contents
   * - Python
     - pybind11 module ``declib._core`` plus the pure-Python ``declib`` package
   * - Operators
     - :cpp:class:`declib::LinearOperator`, :cpp:class:`declib::BilinearOperator`,
       :cpp:class:`declib::NonlinearOperator`, :cpp:class:`declib::Functional`,
       and the concrete DEC operators
   * - Forms
     - :cpp:struct:`declib::FormType`, :cpp:class:`declib::Bundle`,
       :cpp:class:`declib::FormSpace`, :cpp:class:`declib::Form`
   * - Discrete manifold
     - :cpp:class:`declib::DiscreteManifold` = :cpp:class:`declib::Topology` +
       :cpp:class:`declib::GridMapping` + :cpp:class:`declib::Geometry`
   * - Backends
     - ``declib::stag`` (``DMStag``, stencil kernels) now; ``declib::plex`` later
   * - External
     - PETSc, Kokkos, MPI

Principles
----------

* Topology, grid mapping and geometry are separate objects. Metric-free
  operators (:math:`d`, the duality pairing) read no geometry, so their exact
  identities cannot be broken by a geometry change.
* A form's location follows from its type through the mapping and is never
  stored separately.
* Virtual dispatch happens per operator application. Kernels are Kokkos
  functors templated on the dimension, with no virtual calls on device.
* Every linear operator is also a PETSc ``Mat`` (shell or assembled), so
  ``KSP``, ``SNES`` and ``TS`` work unchanged.
* Gradients of functionals and pairing adjoints live in the dual space
  (:cpp:func:`declib::FormSpace::dual_space`), which keeps brackets built from
  :math:`d`, :math:`\star`, :math:`\wedge`, :math:`i_X` and
  :math:`i_X^\dagger` antisymmetric by construction.

Decisions
---------

These were settled on 23 October 2026.

Runtime form types
   Degree and orientation are runtime values in :cpp:struct:`declib::FormType`,
   checked when an operator is built. The spatial dimension stays a template
   parameter in the structured backend.

Extended dual boundary
   :cpp:enumerator:`declib::DualBoundary::Extended` is the default for
   non-periodic grids.

Boundary conditions
   Essential conditions are constrained dofs on the form space
   (:cpp:struct:`declib::EssentialCondition`); natural conditions enter through
   the extended dual boundary cells
   (:cpp:func:`declib::Form::set_boundary_values`).

Wedge and interior product schemes
   Deferred. :cpp:class:`declib::Wedge` and :cpp:class:`declib::InteriorProduct`
   take a :cpp:class:`declib::WedgeScheme` or :cpp:class:`declib::InteriorScheme`
   as a required argument, with no default.

Bindings
   pybind11.

Scalar type
   :cpp:type:`declib::Scalar` is ``PetscScalar``; only real PETSc builds are
   supported.

Invariants
----------

Each class lists the identities it guarantees in its reference
documentation, and the test suite is built from those lists:
:math:`\partial\partial = 0`, :math:`d \circ d = 0`, exact Stokes,
:math:`\star\star = (-1)^{k(n-k)}` for the diagonal star, exact adjointness of
:math:`i_X` and :math:`i_X^\dagger`, agreement of ``apply``, ``shell`` and
``assemble``, and independence of results from the MPI decomposition.

Roadmap
-------

.. list-table::
   :header-rows: 1
   :widths: 10 55 35

   * - Phase
     - Scope
     - Done when
   * - 1
     - Core types, structured topology (1D–3D), grid mapping, uniform geometry,
       scalar forms, :math:`d`, diagonal :math:`\star`, pairing, linear operator
       algebra, PETSc interop, Python bindings
     - Hodge Laplacian solves converge at second order on CPU and GPU, both
       mappings. *Implemented; verified on CPU (OpenMP), GPU pending.*
   * - 2
     - Bilinear and nonlinear frameworks, functionals, wedge and interior
       product with the first schemes, Lie derivative, trivial-bundle forms
     - A conservative advection test conserves discrete energy to round-off
   * - 3
     - Rectilinear and mapped geometry, Galerkin :math:`\star`, connections and
       :math:`d^\nabla`
     - Convergence on a curvilinear grid
   * - 4
     - ``DMPlex`` backend
     - Phase 1 tests pass on simplicial meshes
