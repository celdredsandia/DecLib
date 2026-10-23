Performance portability (Kokkos)
================================

Cochain storage is a PETSc ``Vec``; kernels access it through Kokkos views
(``VecGetKokkosView``), so the same operator code runs on the Serial, OpenMP,
CUDA, HIP and SYCL backends. Select the backend when you build Kokkos and
PETSc (for example ``-vec_type kokkos -mat_type aijkokkos`` at run time).

.. todo:: Add a table of tested backends and benchmark results.
