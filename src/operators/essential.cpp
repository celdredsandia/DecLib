#include "declib/operators/essential.hpp"

#include "declib/core/error.hpp"
#include "detail/petsc.hpp"

namespace declib {

void apply_essential(const FormSpace& space, Mat A, Form& x, Form& b)
{
    if (!x.space().same_type_as(space) || !b.space().same_type_as(space))
        throw TypeError("apply_essential: x and b must belong to the constrained space");
    for (const auto& cond : space.essential_conditions()) {
        IS is = space.boundary_indices(cond.region);
        if (cond.value) {
            // Prescribed values: the de Rham map of the condition's function.
            Form g = space.from_function(*cond.value);
            Vec sub;
            DECLIB_PETSC(VecGetSubVector(g.global_vec(), is, &sub));
            DECLIB_PETSC(VecISCopy(x.global_vec(), is, SCATTER_FORWARD, sub));
            DECLIB_PETSC(VecRestoreSubVector(g.global_vec(), is, &sub));
        } else {
            DECLIB_PETSC(VecISSet(x.global_vec(), is, 0));
        }
        // Keeps A symmetric: zero rows and columns, unit diagonal, and move
        // the known values to the right-hand side.
        DECLIB_PETSC(MatZeroRowsColumnsIS(A, is, 1, x.global_vec(), b.global_vec()));
        DECLIB_PETSC(ISDestroy(&is));
    }
}

} // namespace declib
