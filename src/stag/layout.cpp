#include "declib/stag/detail/layout.hpp"

#include <bit>

#include "detail/petsc.hpp"

namespace declib::stag::detail {

LocalLayout local_layout(DM dm)
{
    LocalLayout l;
    PetscInt dim;
    PetscInt g[3] = {0, 0, 0}, n[3] = {1, 1, 1}, epe;
    DECLIB_PETSC(DMGetDimension(dm, &dim));
    DECLIB_PETSC(DMStagGetGhostCorners(dm, &g[0], &g[1], &g[2], &n[0], &n[1], &n[2]));
    DECLIB_PETSC(DMStagGetEntriesPerElement(dm, &epe));
    for (int a = 0; a < 3; ++a) {
        l.g0[a] = a < dim ? static_cast<int>(g[a]) : 0;
        l.gn[a] = a < dim ? static_cast<int>(n[a]) : 1;
    }
    l.epe = static_cast<int>(epe);
    return l;
}

Box owned_cells(DM dm, unsigned axes)
{
    Box b;
    PetscInt dim;
    PetscInt x[3] = {0, 0, 0}, m[3] = {1, 1, 1}, extra[3] = {0, 0, 0};
    DECLIB_PETSC(DMGetDimension(dm, &dim));
    DECLIB_PETSC(DMStagGetCorners(dm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], &extra[0], &extra[1], &extra[2]));
    for (int a = 0; a < dim; ++a) {
        b.lo[a] = static_cast<int>(x[a]);
        b.hi[a] = static_cast<int>(x[a] + m[a]);
        if (!((axes >> a) & 1u)) b.hi[a] += static_cast<int>(extra[a]);
    }
    return b;
}

DMStagStencilLocation location_of(unsigned axes, int dim)
{
    // Low axes: the cell sits on the low plane of the element along them.
    const unsigned low = ~axes & ((1u << dim) - 1u);
    switch (dim) {
    case 1: return low ? DMSTAG_LEFT : DMSTAG_ELEMENT;
    case 2:
        switch (low) {
        case 3: return DMSTAG_DOWN_LEFT;
        case 2: return DMSTAG_DOWN;
        case 1: return DMSTAG_LEFT;
        default: return DMSTAG_ELEMENT;
        }
    default:
        switch (low) {
        case 7: return DMSTAG_BACK_DOWN_LEFT;
        case 6: return DMSTAG_BACK_DOWN;
        case 5: return DMSTAG_BACK_LEFT;
        case 3: return DMSTAG_DOWN_LEFT;
        case 4: return DMSTAG_BACK;
        case 2: return DMSTAG_DOWN;
        case 1: return DMSTAG_LEFT;
        default: return DMSTAG_ELEMENT;
        }
    }
}

int stratum_index(unsigned axes, int dim)
{
    // Strata are ordered by dimension, then by mask.
    int idx = 0;
    const int k = std::popcount(axes);
    for (unsigned m = 0; m < (1u << dim); ++m) {
        const int km = std::popcount(m);
        if (km < k || (km == k && m < axes)) ++idx;
    }
    return idx;
}

Slots slots_of(DM dm, int dim)
{
    Slots s;
    for (auto& row : s.s)
        for (int& v : row) v = -1;
    PetscInt dof[4] = {0, 0, 0, 0};
    DECLIB_PETSC(DMStagGetDOF(dm, &dof[0], &dof[1], &dof[2], &dof[3]));
    for (unsigned axes = 0; axes < (1u << dim); ++axes) {
        const int k = std::popcount(axes);
        const int si = stratum_index(axes, dim);
        if (dof[k] > max_components) throw Error("fiber rank exceeds the supported maximum");
        for (PetscInt c = 0; c < dof[k]; ++c) {
            PetscInt slot;
            DECLIB_PETSC(DMStagGetLocationSlot(dm, location_of(axes, dim), c, &slot));
            s.s[si][c] = static_cast<int>(slot);
        }
    }
    return s;
}

} // namespace declib::stag::detail
