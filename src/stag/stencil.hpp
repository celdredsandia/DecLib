// Internal: structured stencil operators (exterior derivatives, diagonal
// Hodge stars) applied with Kokkos kernels or assembled into PETSc matrices.
#pragma once

#include <vector>

#include <petscdmstag.h>
#include <petscmat.h>

#include "declib/core/types.hpp"
#include "declib/forms/form.hpp"

namespace declib::detail {

// y(cell) += coef * x(src stratum, cell + off), per fiber component.
struct StencilTerm {
    int src = 0;          // stratum index of the source cell
    int off[3] = {0, 0, 0};
    Scalar coef = 0;
};

// All terms producing values on one destination stratum.
struct StencilRow {
    unsigned dst_axes = 0;
    int dst = 0; // stratum index
    std::vector<StencilTerm> terms;
};

struct StencilOp {
    int dim = 0;
    int ncomp = 1;
    std::vector<StencilRow> rows;
    // Optional pointwise factor num/den on destination cells. Both are ghosted
    // local vectors (not owned) on DMs with one dof on the destination strata.
    Vec scale_num = nullptr;
    Vec scale_den = nullptr;

    // Compute owned entries of the local vector `yloc` from the ghosted local `xloc`.
    void apply_local(DM xdm, Vec xloc, DM ydm, Vec yloc) const;

    // y = S x on global forms.
    void apply(const Form& x, Form& y) const;

    // Assembled matrix with rows in the layout of `ydm` and columns of `xdm`.
    Mat assemble(DM xdm, DM ydm, MatType type) const;
};

// d on k-cochains hosted on `host`, with `ncomp` fiber components.
StencilOp exterior_derivative_stencil(int n, Complex host, int k, int ncomp);

// Signs of the diagonal Hodge star on k-forms hosted on `host` (scale fields
// are attached by the caller).
StencilOp hodge_stencil(int n, Complex host, int k, int ncomp);

} // namespace declib::detail
