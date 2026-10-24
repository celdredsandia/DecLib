# DecLib
A C++ library for discrete exterior calculus on PETSc and Kokkos, with a
Python interface.

Phase 1 is implemented: structured grids (DMStag) in 1D–3D, straight and
twisted forms on either grid mapping, the exterior derivative, the diagonal
Hodge star, the duality pairing, linear operator algebra with PETSc `Mat`/`KSP`
interop, essential boundary conditions, and pybind11 bindings. The Hodge
Laplacian converges at second order for every degree in 1D–3D.

## Building

```bash
export PKG_CONFIG_PATH=$PETSC_PREFIX/lib/pkgconfig      # PETSc 3.22 built with Kokkos
export Kokkos_DIR=$PETSC_PREFIX/lib/cmake/Kokkos
cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DDECLIB_BUILD_PYTHON=ON
cmake --build build -j && ctest --test-dir build --output-on-failure
```

See `docs/getting-started/installation.rst` for building PETSc and the options.

## Documentation

The docs combine Doxygen + Breathe + Exhale (C++ API) with Sphinx autodoc
(Python API).

```bash
sudo apt-get install doxygen graphviz
pip install -r docs/requirements.txt
make -C docs html          # or: make -C docs strict  (warnings are errors, as in CI)
# or via CMake:
cmake -S . -B build-docs -DDECLIB_BUILD_LIBRARY=OFF -DDECLIB_BUILD_DOCS=ON && cmake --build build-docs --target docs
```

Set `DECLIB_DOCS_OFFLINE=1` to build without network access (skips intersphinx).
