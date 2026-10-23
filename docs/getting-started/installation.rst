Installation
============

Requirements
------------

========================  ==================  ===============================
Dependency                Minimum version     Notes
========================  ==================  ===============================
C++ compiler              C++20               GCC 11+, Clang 14+, NVCC 12+
CMake                     3.24
PETSc                     3.22                Built with MPI
Kokkos                    4.4                 Any backend (Serial, OpenMP,
                                              CUDA, HIP, SYCL)
Python *(optional)*       3.10                For the Python bindings
petsc4py *(optional)*     matches PETSc
========================  ==================  ===============================

Building the C++ library
------------------------

.. code-block:: bash

   cmake -S . -B build \
         -DCMAKE_BUILD_TYPE=Release \
         -DPETSC_DIR=$PETSC_DIR -DPETSC_ARCH=$PETSC_ARCH \
         -DKokkos_ROOT=/path/to/kokkos
   cmake --build build -j
   ctest --test-dir build

Installing the Python package
-----------------------------

.. code-block:: bash

   pip install ./python

Building this documentation
---------------------------

See :doc:`/developer/writing-docs`.
