Performance portability (Kokkos)
================================

Form values live in PETSc ``Vec`` objects of type ``VECKOKKOS``; kernels access
them through Kokkos views (``VecGetKokkosView``), so the same operator code
runs on the Serial, OpenMP, CUDA, HIP and SYCL backends. Virtual dispatch
happens once per operator application, never inside a kernel.

Every :cpp:class:`declib::LinearOperator` can present itself to PETSc either
matrix-free (:cpp:func:`declib::LinearOperator::shell`) or assembled
(:cpp:func:`declib::LinearOperator::assemble`, ``MATAIJKOKKOS`` by default,
built on device with ``MatSetValuesCOO``).

.. todo:: Add a table of tested backends and benchmark results.
