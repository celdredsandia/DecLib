// Internal: PETSc error handling and small RAII helpers.
#pragma once

#include <string>

#include <petscsys.h>

#include "declib/core/error.hpp"

namespace declib::detail {

[[noreturn]] void throw_petsc(PetscErrorCode code, const char* call, const char* file, int line);

// Make sure Kokkos is initialized (PETSc initializes it on first use of a
// Kokkos object if the application did not).
void ensure_kokkos();

} // namespace declib::detail

#define DECLIB_PETSC(call)                                                          \
    do {                                                                            \
        PetscErrorCode declib_ierr_ = (call);                                       \
        if (declib_ierr_) ::declib::detail::throw_petsc(declib_ierr_, #call, __FILE__, __LINE__); \
    } while (0)
