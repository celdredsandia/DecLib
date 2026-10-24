# AGENTS.md

Guidance for coding agents working on DecLib++. Read `README.md` for what the
library does and `docs/developer/architecture.rst` for how it is put together.

## Project at a glance

- C++20 library for discrete exterior calculus on PETSc (data, layouts,
  solvers) and Kokkos (kernels), with a Python interface through pybind11.
- Structured grids use PETSc `DMStag` (namespace `declib::stag`). Unstructured
  grids (`DMPlex`) come in phase 4; keep backend-specific code out of the
  generic layers (`core`, `topology`, `mapping`, `geometry`, `manifold`,
  `forms`, `operators`).
- Phase 1 is implemented. Wedge, interior product, Lie derivative, covariant
  derivative, non-diagonal Hodge stars and nonzero natural boundary data
  throw on purpose until their phase. Do not remove those errors without
  implementing the feature.

## Layout

| Path | What lives there |
|---|---|
| `include/declib/` | Public headers, each fully documented with Doxygen |
| `include/declib/stag/detail/` | Internal layout helpers shared by sources and `FormView` |
| `src/` | Implementations; `src/detail/petsc.hpp` has `DECLIB_PETSC` |
| `src/stag/stencil.*` | Stencil engine for d and the diagonal star (Kokkos apply, COO assembly) |
| `python/src/core.cpp` | pybind11 extension `declib._core` (thin handles) |
| `python/declib/` | Pure-Python API mirroring the C++ classes |
| `tests/`, `python/tests/` | GoogleTest and pytest suites |
| `examples/` | Quickstarts, run as tests and included in the docs |
| `docs/` | Sphinx sources; `docs/design/` holds the original design document |

## Build and test

Needs PETSc 3.22 configured with MPI, real scalars and Kokkos
(`--with-kokkos=1 --download-kokkos=1 --download-kokkos-kernels=1`). Do not
pass `--with-cxx-dialect=20` to PETSc; its Kokkos is capped at C++17, which is
fine. For Python, petsc4py 3.22 needs `cython<3.1` and `setuptools<80`.

```bash
export PKG_CONFIG_PATH=$PETSC_PREFIX/lib/pkgconfig
export Kokkos_DIR=$PETSC_PREFIX/lib/cmake/Kokkos
cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
      -DDECLIB_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python) \
      -DDECLIB_MPIEXEC_PREFLAGS=--oversubscribe
cmake --build build -j
ctest --test-dir build --output-on-failure      # C++ on 1, 2, 4 ranks + pytest + examples
```

- Run one C++ test: `./build/tests/declib_tests --gtest_filter='HodgeStar.*'`.
- Run pytest directly: `PYTHONPATH=python:$PETSC_PREFIX/lib python -m pytest python/tests`.
- Set `OMP_NUM_THREADS=1` when running several MPI ranks on a small machine.
  As root, OpenMPI also needs `OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`.
- If GoogleTest cannot be downloaded, install it (`libgtest-dev`); CMake
  prefers an installed copy.
- Docs only: `cmake -S . -B build-docs -DDECLIB_BUILD_LIBRARY=OFF -DDECLIB_BUILD_DOCS=ON`
  then `cmake --build build-docs --target docs`. Add `DECLIB_DOCS_OFFLINE=1`
  without network access.

Before pushing, the full CTest run and the strict docs build must both pass.
CI (`.github/workflows/ci.yml`, `docs.yml`) runs the same commands.

## Conventions

- **Documentation is part of the API.** Every public class, function and
  member gets a Doxygen comment, and the docs build runs with warnings as
  errors (`nitpicky = True`). When you change a public interface, update its
  Python mirror, the Sphinx pages that describe it, and the examples.
- **Signs come from one place.** Orientation conventions live in
  `include/declib/stag/orientation.hpp`: cells are oriented by their sorted
  axes, dual cells by their sorted complementary axes, and strata are numbered
  by dimension and then by axis bit mask. The stencils for d, the Hodge star
  and the pairing all derive their signs from it; do not hard-code signs
  elsewhere.
- **Dual cells are indexed by primal cells.** A dual k-cell is stored on the
  stratum of the primal (n−k)-cell it crosses. `GridMapping::storage_cells`
  gives the strata for a form type.
- **PETSc calls** in library code are wrapped in `DECLIB_PETSC(...)`, which
  throws `declib::PetscException`. Do not name anything `PetscError` inside
  `namespace declib`; it hides PETSc's function and breaks `PetscCall`. PETSc
  callbacks that use `PetscCall`/`SETERRQ` live outside `namespace declib`.
- **Header order:** a source file that uses `VecGetKokkosView` must include
  `<petscvec_kokkos.hpp>` before any other PETSc header.
- **Type errors** throw `declib::TypeError` when an operator is built, not when
  it is applied. Python sees them as `TypeError`.
- **Ownership:** factories return `std::shared_ptr<const T>`; spaces hold their
  manifold, and the manifold caches spaces weakly. Functions that return a
  PETSc object (`assemble`, `shell`, `boundary_indices`) hand ownership to
  the caller.
- **Non-periodic axes** are always `DM_BOUNDARY_GHOSTED`. Local vectors start
  at zero, so reads outside the domain see zero (homogeneous natural data).
- **Style:** match the surrounding code: 4-space indent, braces on the next
  line for functions, `snake_case` functions, `PascalCase` types, comments
  only where the reason is not obvious.

## Tests

- Every change comes with tests. Prefer checking a discrete identity
  (d∘d = 0, Stokes, ⋆⋆ = ±1, adjointness, apply/assemble/shell agreement)
  over comparing numbers.
- C++ tests run on 1, 2 and 4 ranks, so they must be correct in parallel and
  use grids with at least 4 cells per axis.
- Loop new tests over dimensions 1–3, both grid mappings, both orientations,
  and periodic and bounded grids where that makes sense.
- Never weaken, skip or delete a test to make CI pass.

## Git

- Work on a feature branch and open a pull request against `main`.
- Keep commits focused, with a message that says what changed and why.
