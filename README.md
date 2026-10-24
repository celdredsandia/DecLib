# DecLib++

A C++ library for discrete exterior calculus (DEC) on
[PETSc](https://petsc.org) and [Kokkos](https://kokkos.org), with a Python
interface.

DecLib++ represents differential forms as cochains on a cell complex and its
dual, and builds the operators of exterior calculus on them: the exterior
derivative, the Hodge star, the duality pairing, and operators derived from
them such as the codifferential and the Hodge Laplacian. Form data lives in
PETSc vectors, kernels are written with Kokkos for CPUs and GPUs, and every
linear operator can be handed to PETSc as a matrix-free shell or an assembled
sparse matrix, so `KSP`, `SNES` and `TS` work on it directly.

## Status

Phase 1 of four is implemented and tested on CPU (OpenMP backend):

- Topologically structured grids in 1D, 2D and 3D on `DMStag`, with periodic
  or non-periodic axes and the dual complex closed by extended boundary cells.
- Straight (inner-oriented) and twisted (outer-oriented) forms, with either
  orientation hosted on the primal grid (`GridMapping::straight_on_primal()`
  or `GridMapping::twisted_on_primal()`).
- Uniform Cartesian geometry: cell measures, coordinates, and the de Rham map
  that integrates a smooth form over cells.
- Operators: exterior derivative, diagonal Hodge star, duality pairing and
  pairing adjoints, codifferential, Hodge Laplacian, mass.
- Operator algebra: composition, sums, scaling, transpose, identity, exact
  diagonal inverses and Krylov (`KSP`) inverses, with type checks when
  operators are built.
- Essential (Dirichlet-type) boundary conditions as constrained degrees of
  freedom.
- Python bindings (pybind11) with petsc4py interop.

The Hodge Laplacian converges at second order for every form degree in 1D–3D
on both grid mappings, and the test suite checks the discrete identities
(d∘d = 0, Stokes, ⋆⋆ = ±1, adjointness) on 1, 2 and 4 MPI ranks.

Not yet implemented: GPU verification, nonzero natural boundary data,
non-diagonal Hodge stars, the wedge and interior products, the Lie derivative,
connections for bundle-valued forms, and unstructured grids. The roadmap is:

| Phase | Scope | Status |
|---|---|---|
| 1 | Structured topology, uniform geometry, scalar forms, d, diagonal ⋆, pairing, operator algebra, PETSc interop, Python | Done (CPU) |
| 2 | Bilinear and nonlinear operators, wedge and interior product, Lie derivative, natural boundary data | Planned |
| 3 | Rectilinear and mapped geometry, Galerkin ⋆, connections and covariant derivative | Planned |
| 4 | Unstructured grids on `DMPlex` | Planned |

## Example

The Laplacian ∇²u = ⋆⁻¹ d ⋆ d u of a 0-form on the periodic unit square:

```cpp
#include <declib/declib.hpp>

using namespace declib;
auto topo = stag::Topology<2>::create(PETSC_COMM_WORLD, {.cells = {64, 64}});
auto geom = stag::UniformGeometry<2>::create(topo, {.lengths = {1.0, 1.0}});
auto M    = DiscreteManifold::create(topo, geom);

auto V0  = M->space(0, Orientation::Straight);   // values on primal vertices
auto V1  = M->space(1, Orientation::Straight);   // primal edges
auto tV1 = M->space(1, Orientation::Twisted);    // dual edges

OpPtr lap = inverse(HodgeStar::create(V0)) * ExteriorDerivative::create(tV1)
          * HodgeStar::create(V1) * ExteriorDerivative::create(V0);

Form u = V0->from_function([](auto x, auto v) { v[0] = std::sin(2 * M_PI * x[0]); });
Form f = (*lap)(u);          // apply matrix-free
Mat  A = lap->assemble();    // or as a PETSc matrix
```

```python
import declib as dl

topo = dl.StagTopology((64, 64))
M = dl.DiscreteManifold(topo, dl.UniformGeometry(topo))
S, T = dl.Orientation.STRAIGHT, dl.Orientation.TWISTED
V0, V1, tV1 = M.space(0, S), M.space(1, S), M.space(1, T)

lap = (dl.inverse(dl.HodgeStar(V0)) @ dl.ExteriorDerivative(tV1)
       @ dl.HodgeStar(V1) @ dl.ExteriorDerivative(V0))
A = lap.assemble()           # petsc4py.PETSc.Mat
```

Complete programs are in [`examples/`](examples/) and run as part of the tests.

## Concepts

- **`DiscreteManifold`** joins a `Topology` (the cell complex and its dual), a
  `GridMapping` (which complex hosts straight forms and which twisted ones)
  and a `Geometry` (measures and coordinates). Metric-free operators such as d
  and the pairing never read the geometry.
- **`FormSpace`** is the space of forms of one `FormType` (degree,
  orientation, bundle). Spaces are cached by the manifold; each has its own
  PETSc `DM` layout.
- **`Form`** holds the values: a `VECKOKKOS` vector plus a ghosted local copy
  for kernels.
- **`LinearOperator`** maps one space to another. It can apply itself, its
  transpose and its adjoint under the duality pairing, and present itself to
  PETSc as a `MATSHELL` or an assembled matrix.

The design and the reasons behind it are in
[`docs/design/declibpp-design.html`](docs/design/declibpp-design.html) and
[`docs/developer/architecture.rst`](docs/developer/architecture.rst).

## Repository layout

| Path | Contents |
|---|---|
| `include/declib/` | Public headers: `core`, `topology`, `mapping`, `geometry`, `manifold`, `forms`, `operators`, and the structured backend `stag` |
| `src/` | Implementation; `src/stag/stencil.*` holds the Kokkos stencil engine |
| `python/declib/` | Python package (pure Python, wrapping the extension) |
| `python/src/core.cpp` | pybind11 extension `declib._core` |
| `tests/`, `python/tests/` | GoogleTest and pytest suites |
| `examples/` | Quickstart programs in C++ and Python |
| `docs/` | Sphinx documentation, the design document, Doxygen config |
| `.github/workflows/` | CI (build and test) and docs (build and deploy) |

## Building

Requirements: a C++20 compiler, CMake 3.24+, MPI, and PETSc 3.22 configured
with Kokkos and real scalars. For Python: Python 3.10+, petsc4py and pybind11.

Build PETSc with Kokkos (petsc4py is optional):

```bash
./configure --prefix=$HOME/petsc --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
    --with-debugging=0 --with-openmp=1 \
    --with-kokkos=1 --download-kokkos=1 --download-kokkos-kernels=1 --with-petsc4py=1
make && make install
```

Then build and test DecLib++:

```bash
export PKG_CONFIG_PATH=$HOME/petsc/lib/pkgconfig
export Kokkos_DIR=$HOME/petsc/lib/cmake/Kokkos
cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
      -DDECLIB_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python)
cmake --build build -j
PYTHONPATH=$HOME/petsc/lib ctest --test-dir build --output-on-failure
```

The Python package then works in place with
`PYTHONPATH=$PWD/python:$HOME/petsc/lib`. See
[`docs/getting-started/installation.rst`](docs/getting-started/installation.rst)
for all CMake options and version notes (for example, petsc4py 3.22 needs
Cython older than 3.1).

## Documentation

The documentation combines Doxygen, Breathe and Exhale for the C++ API with
Sphinx autodoc for the Python API. CI builds it with warnings treated as errors
and deploys it to GitHub Pages from `main`.

```bash
sudo apt-get install doxygen graphviz
pip install -r docs/requirements.txt
cmake -S . -B build-docs -DDECLIB_BUILD_LIBRARY=OFF -DDECLIB_BUILD_DOCS=ON
cmake --build build-docs --target docs        # output in build-docs/docs/html
```

Set `DECLIB_DOCS_OFFLINE=1` to build without network access (skips intersphinx).

## Contributing

See [`docs/developer/contributing.rst`](docs/developer/contributing.rst) and
[`docs/developer/testing.rst`](docs/developer/testing.rst). Coding agents
should read [`AGENTS.md`](AGENTS.md) first.
