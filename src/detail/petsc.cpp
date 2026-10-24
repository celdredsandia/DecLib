#include "detail/petsc.hpp"

#include <sstream>

#include <Kokkos_Core.hpp>
#include <petscvec.h>

namespace declib::detail {

void throw_petsc(PetscErrorCode code, const char* call, const char* file, int line)
{
    const char* text = nullptr;
    PetscErrorMessage(code, &text, nullptr);
    std::ostringstream msg;
    msg << "PETSc error " << code << " (" << (text ? text : "unknown") << ") in " << call << " at " << file
        << ":" << line;
    throw PetscException(msg.str());
}

void ensure_kokkos()
{
    if (Kokkos::is_initialized()) return;
    // Creating a Kokkos vector makes PETSc initialize Kokkos (and finalize it
    // in PetscFinalize).
    Vec v;
    DECLIB_PETSC(VecCreate(PETSC_COMM_SELF, &v));
    DECLIB_PETSC(VecSetSizes(v, 1, 1));
    DECLIB_PETSC(VecSetType(v, VECKOKKOS));
    DECLIB_PETSC(VecDestroy(&v));
}

} // namespace declib::detail
