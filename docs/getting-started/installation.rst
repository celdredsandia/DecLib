Installation
============

Requirements
------------

========================  ==================  ===============================
Dependency                Version             Notes
========================  ==================  ===============================
C++ compiler              C++20               GCC 11+, Clang 14+, NVCC 12+
CMake                     3.24
PETSc                     3.22                Built with MPI, Kokkos and
                                              real scalars (other builds are
                                              rejected at configure time)
Kokkos                    4.4                 Usually downloaded by PETSc;
                                              any backend
Python *(optional)*       3.10                For the Python bindings
petsc4py *(optional)*     matches PETSc
pybind11 *(optional)*     2.12                Builds the ``declib._core``
                                              extension
GoogleTest *(optional)*   1.11                Used if installed, otherwise
                                              downloaded
========================  ==================  ===============================

Building PETSc
--------------

DecLib++ needs PETSc configured with Kokkos. Letting PETSc download Kokkos and
Kokkos Kernels keeps the versions consistent:

.. code-block:: bash

   ./configure --prefix=$HOME/petsc \
       --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
       --with-debugging=0 --with-openmp=1 \
       --with-kokkos=1 --download-kokkos=1 --download-kokkos-kernels=1 \
       --with-petsc4py=1        # optional, for the Python bindings
   make && make install

Add ``--with-cuda=1`` or ``--with-hip=1`` for GPUs. Do not pass
``--with-cxx-dialect=20``: PETSc 3.22 builds Kokkos as C++17, which is
compatible with DecLib++ compiled as C++20. Building petsc4py 3.22 needs
Cython older than 3.1.

Building DecLib++
-----------------

PETSc is found with ``pkg-config`` and Kokkos with its CMake package:

.. code-block:: bash

   export PKG_CONFIG_PATH=$HOME/petsc/lib/pkgconfig
   export Kokkos_DIR=$HOME/petsc/lib/cmake/Kokkos
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc
   cmake --build build -j
   ctest --test-dir build --output-on-failure

CMake options:

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Option
     - Default
     - Effect
   * - ``DECLIB_BUILD_LIBRARY``
     - ON
     - The C++ library; OFF builds the documentation only
   * - ``DECLIB_BUILD_TESTS``
     - ON
     - C++ tests, run on 1, 2 and 4 MPI ranks
   * - ``DECLIB_BUILD_EXAMPLES``
     - ON
     - Programs in ``examples/``
   * - ``DECLIB_BUILD_PYTHON``
     - OFF
     - The extension ``declib._core``
   * - ``DECLIB_BUILD_DOCS``
     - OFF
     - This documentation
   * - ``DECLIB_MPIEXEC_PREFLAGS``
     - empty
     - Extra ``mpiexec`` flags for the tests, e.g. ``--oversubscribe``

Python bindings
---------------

Configure with ``-DDECLIB_BUILD_PYTHON=ON`` and the Python interpreter that
has petsc4py and pybind11 (``-DPython_EXECUTABLE=...``). The extension is
written next to the package sources, so the package works in place:

.. code-block:: bash

   export PYTHONPATH=$PWD/python:$HOME/petsc/lib   # petsc4py from --with-petsc4py
   python -c "import declib"
   python -m pytest python/tests

Building this documentation
---------------------------

See :doc:`/developer/writing-docs`.
