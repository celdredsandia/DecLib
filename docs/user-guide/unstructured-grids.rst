Unstructured grids (DMPlex)
===========================

Unstructured grids are planned but not yet implemented. Nothing above the
backend layer refers to ``DMStag``, so a ``DMPlex`` backend adds
implementations rather than changing interfaces:

.. list-table::
   :header-rows: 1

   * - Interface
     - Structured (now)
     - Unstructured (planned)
   * - :cpp:class:`declib::Topology`
     - :cpp:class:`declib::stag::Topology`, implicit incidence
     - cones from the ``DMPlex`` Hasse diagram, CSR incidence on device
   * - :cpp:class:`declib::Geometry`
     - uniform, rectilinear, mapped
     - simplicial or polyhedral, circumcentric or barycentric duals
   * - :cpp:class:`declib::FormSpace`
     - compatible ``DMStag`` per form type
     - ``DMPlex`` clone with a ``PetscSection`` on one depth stratum
   * - kernels
     - stencils over ``MDRangePolicy``
     - sparse loops over incidence

Dual cells are indexed by their primal partner on both backends, and every
operator has a generic path that applies its assembled matrix, so a new
backend runs end to end as soon as :cpp:func:`declib::Topology::incidence`
and the geometric measures exist.
