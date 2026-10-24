Testing
=======

C++ tests run with CTest (``ctest --test-dir build``) and Python tests with
``pytest python/tests`` (also registered with CTest when the bindings are
built).

C++ tests
---------

The suite in ``tests/`` uses GoogleTest with a driver that initializes PETSc
and fails the run if any rank fails. CTest runs the same executable on 1, 2
and 4 MPI ranks; pass ``-DDECLIB_MPIEXEC_PREFLAGS=--oversubscribe`` on
machines with fewer cores.

``test_topology.cpp``
   Cell counts with and without periodicity, strata order and ``DMStag``
   locations, incidence matrices as transposes of :math:`d`.
``test_forms.cpp``
   Vector operations, ghost refresh and ``commit_local``, ``FormView``
   indexing, the de Rham map of constants, boundary indices and the
   validation of essential conditions.
``test_exterior_derivative.cpp``
   :math:`d \circ d = 0`, agreement of ``apply``, ``assemble`` and
   ``shell``, :math:`d R = R d` (Stokes), and the structured pairing adjoint
   against the generic one, for :math:`n = 1, 2, 3`, all degrees, both
   mappings, both orientations, periodic and bounded grids.
``test_hodge_star.cpp``
   :math:`\star\star = (-1)^{k(n-k)}`, positivity of
   :math:`\langle a, \star a \rangle`, half cells at the boundary.
``test_operators.cpp``
   Composite operators (codifferential, Hodge Laplacian, sums, transposes)
   against their assembled matrices and transposes, Krylov inverses, and
   results that do not depend on the MPI decomposition.
``test_convergence.cpp``
   The phase 1 acceptance tests: second-order consistency of the Hodge
   Laplacian for every degree in 1D–3D on both mappings, and second-order
   convergence of Poisson solves with essential conditions (primal) and
   extended dual boundary cells (dual). Rates are printed in the log.

Python tests
------------

``python/tests/test_declib.py`` covers the same invariants through the
Python interface, plus petsc4py interop and a Poisson solve with
``petsc4py.PETSc.KSP``.

Continuous integration
----------------------

The ``CI`` workflow builds PETSc with Kokkos (OpenMP backend) and petsc4py,
caches the installation, then builds DecLib++ with the bindings and runs
CTest. GPU backends are not tested in CI yet.
