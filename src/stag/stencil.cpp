// petscvec_kokkos.hpp must precede every other PETSc header.
#include <petscvec_kokkos.hpp>

#include "stag/stencil.hpp"

#include <bit>


#include "declib/stag/detail/layout.hpp"
#include "declib/stag/orientation.hpp"
#include "detail/petsc.hpp"

namespace declib::detail {

namespace {

constexpr int max_terms = 6;

// Device copy of one StencilRow with slots resolved.
struct RowData {
    int nterms = 0;
    int off[max_terms][3] = {};
    Scalar coef[max_terms] = {};
    int xslot[max_terms][stag::detail::max_components] = {};
    int yslot[stag::detail::max_components] = {};
    int sslot = -1;
};

RowData row_data(const StencilRow& row, int ncomp, const stag::detail::Slots& xs, const stag::detail::Slots& ys,
                 const stag::detail::Slots* ss)
{
    RowData r;
    if (row.terms.size() > static_cast<std::size_t>(max_terms)) throw Error("stencil has too many terms");
    r.nterms = static_cast<int>(row.terms.size());
    for (int t = 0; t < r.nterms; ++t) {
        for (int a = 0; a < 3; ++a) r.off[t][a] = row.terms[t].off[a];
        r.coef[t] = row.terms[t].coef;
        for (int c = 0; c < ncomp; ++c) r.xslot[t][c] = xs.s[row.terms[t].src][c];
    }
    for (int c = 0; c < ncomp; ++c) r.yslot[c] = ys.s[row.dst][c];
    if (ss) r.sslot = ss->s[row.dst][0];
    return r;
}

DM dm_of(Vec v)
{
    DM dm;
    DECLIB_PETSC(VecGetDM(v, &dm));
    return dm;
}

PetscInt global_local_size(DM dm)
{
    Vec g;
    PetscInt n;
    DECLIB_PETSC(DMGetGlobalVector(dm, &g));
    DECLIB_PETSC(VecGetLocalSize(g, &n));
    DECLIB_PETSC(DMRestoreGlobalVector(dm, &g));
    return n;
}

} // namespace

void StencilOp::apply_local(DM xdm, Vec xloc, DM ydm, Vec yloc) const
{
    using ConstView = Kokkos::View<const PetscScalar*, MemSpace>;
    using View = Kokkos::View<PetscScalar*, MemSpace>;
    const auto xl = stag::detail::local_layout(xdm);
    const auto yl = stag::detail::local_layout(ydm);
    const auto xs = stag::detail::slots_of(xdm, dim);
    const auto ys = stag::detail::slots_of(ydm, dim);
    const bool scaled = scale_num != nullptr;
    stag::detail::LocalLayout sl;
    stag::detail::Slots ss{};
    ConstView num, den;
    if (scaled) {
        sl = stag::detail::local_layout(dm_of(scale_num));
        ss = stag::detail::slots_of(dm_of(scale_num), dim);
        DECLIB_PETSC(VecGetKokkosView(scale_num, &num));
        DECLIB_PETSC(VecGetKokkosView(scale_den, &den));
    }
    ConstView x;
    View y;
    DECLIB_PETSC(VecGetKokkosView(xloc, &x));
    DECLIB_PETSC(VecGetKokkosView(yloc, &y));
    const int nc = ncomp;
    for (const auto& row : rows) {
        const RowData r = row_data(row, nc, xs, ys, scaled ? &ss : nullptr);
        const auto box = stag::detail::owned_cells(ydm, row.dst_axes);
        if (box.size() == 0) continue;
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>> policy({box.lo[0], box.lo[1], box.lo[2]},
                                                                 {box.hi[0], box.hi[1], box.hi[2]});
        Kokkos::parallel_for(
            "declib::stencil", policy, KOKKOS_LAMBDA(int i, int j, int k) {
                for (int c = 0; c < nc; ++c) {
                    Scalar acc = 0;
                    for (int t = 0; t < r.nterms; ++t)
                        acc += r.coef[t] *
                               x(xl.index(i + r.off[t][0], j + r.off[t][1], k + r.off[t][2], r.xslot[t][c]));
                    if (scaled) acc *= num(sl.index(i, j, k, r.sslot)) / den(sl.index(i, j, k, r.sslot));
                    y(yl.index(i, j, k, r.yslot[c])) = acc;
                }
            });
    }
    Kokkos::fence();
    DECLIB_PETSC(VecRestoreKokkosView(xloc, &x));
    DECLIB_PETSC(VecRestoreKokkosView(yloc, &y));
    if (scaled) {
        DECLIB_PETSC(VecRestoreKokkosView(scale_num, &num));
        DECLIB_PETSC(VecRestoreKokkosView(scale_den, &den));
    }
}

void StencilOp::apply(const Form& x, Form& y) const
{
    DM xdm = x.space().dm();
    DM ydm = y.space().dm();
    Vec yloc;
    DECLIB_PETSC(DMGetLocalVector(ydm, &yloc));
    apply_local(xdm, x.local_vec(), ydm, yloc);
    DECLIB_PETSC(DMLocalToGlobal(ydm, yloc, INSERT_VALUES, y.global_vec()));
    DECLIB_PETSC(DMRestoreLocalVector(ydm, &yloc));
}

Mat StencilOp::assemble(DM xdm, DM ydm, MatType type) const
{
    const auto xl = stag::detail::local_layout(xdm);
    const auto yl = stag::detail::local_layout(ydm);
    const auto xs = stag::detail::slots_of(xdm, dim);
    const auto ys = stag::detail::slots_of(ydm, dim);
    const PetscScalar* num = nullptr;
    const PetscScalar* den = nullptr;
    stag::detail::LocalLayout sl;
    stag::detail::Slots ss{};
    if (scale_num) {
        sl = stag::detail::local_layout(dm_of(scale_num));
        ss = stag::detail::slots_of(dm_of(scale_num), dim);
        DECLIB_PETSC(VecGetArrayRead(scale_num, &num));
        DECLIB_PETSC(VecGetArrayRead(scale_den, &den));
    }
    std::vector<PetscInt> ri, ci;
    std::vector<PetscScalar> v;
    for (const auto& row : rows) {
        const RowData r = row_data(row, ncomp, xs, ys, scale_num ? &ss : nullptr);
        const auto b = stag::detail::owned_cells(ydm, row.dst_axes);
        for (int k = b.lo[2]; k < b.hi[2]; ++k)
            for (int j = b.lo[1]; j < b.hi[1]; ++j)
                for (int i = b.lo[0]; i < b.hi[0]; ++i) {
                    const Scalar f = num ? num[sl.index(i, j, k, r.sslot)] / den[sl.index(i, j, k, r.sslot)] : 1;
                    for (int c = 0; c < ncomp; ++c)
                        for (int t = 0; t < r.nterms; ++t) {
                            ri.push_back(yl.index(i, j, k, r.yslot[c]));
                            ci.push_back(xl.index(i + r.off[t][0], j + r.off[t][1], k + r.off[t][2], r.xslot[t][c]));
                            v.push_back(r.coef[t] * f);
                        }
                }
    }
    if (scale_num) {
        DECLIB_PETSC(VecRestoreArrayRead(scale_num, &num));
        DECLIB_PETSC(VecRestoreArrayRead(scale_den, &den));
    }
    MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(ydm));
    ISLocalToGlobalMapping rmap, cmap;
    DECLIB_PETSC(DMGetLocalToGlobalMapping(ydm, &rmap));
    DECLIB_PETSC(DMGetLocalToGlobalMapping(xdm, &cmap));
    Mat A;
    DECLIB_PETSC(MatCreate(comm, &A));
    DECLIB_PETSC(MatSetSizes(A, global_local_size(ydm), global_local_size(xdm), PETSC_DETERMINE, PETSC_DETERMINE));
    DECLIB_PETSC(MatSetType(A, type));
    DECLIB_PETSC(MatSetLocalToGlobalMapping(A, rmap, cmap));
    DECLIB_PETSC(MatSetPreallocationCOOLocal(A, static_cast<PetscCount>(v.size()), ri.data(), ci.data()));
    DECLIB_PETSC(MatSetValuesCOO(A, v.data(), INSERT_VALUES));
    return A;
}

StencilOp exterior_derivative_stencil(int n, Complex host, int k, int ncomp)
{
    if (k < 0 || k >= n) throw TypeError("exterior derivative needs 0 <= k < n");
    StencilOp op;
    op.dim = n;
    op.ncomp = ncomp;
    for (unsigned axes = 0; axes < (1u << n); ++axes) {
        StencilRow row;
        row.dst_axes = axes;
        row.dst = stag::detail::stratum_index(axes, n);
        if (host == Complex::Primal) {
            // Range: primal (k+1)-cells T; d w(T) = sum_m (-1)^m [w(T - b_m at +e_b) - w(T - b_m)].
            if (std::popcount(axes) != k + 1) continue;
            for (int b = 0; b < n; ++b) {
                if (!((axes >> b) & 1u)) continue;
                const int s = stag::boundary_sign(axes, b);
                const int src = stag::detail::stratum_index(axes & ~(1u << b), n);
                StencilTerm hi{src, {0, 0, 0}, Scalar(s)};
                hi.off[b] = 1;
                row.terms.push_back(hi);
                row.terms.push_back(StencilTerm{src, {0, 0, 0}, Scalar(-s)});
            }
        } else {
            // Range: dual (k+1)-cells crossing primal (n-k-1)-cells A, spanning B = A^c.
            if (std::popcount(axes) != n - k - 1) continue;
            const unsigned dual_axes = stag::complement(axes, n);
            for (int b = 0; b < n; ++b) {
                if (!((dual_axes >> b) & 1u)) continue;
                const int s = stag::boundary_sign(dual_axes, b);
                const int src = stag::detail::stratum_index(axes | (1u << b), n);
                row.terms.push_back(StencilTerm{src, {0, 0, 0}, Scalar(s)});
                StencilTerm lo{src, {0, 0, 0}, Scalar(-s)};
                lo.off[b] = -1;
                row.terms.push_back(lo);
            }
        }
        op.rows.push_back(std::move(row));
    }
    return op;
}

StencilOp hodge_stencil(int n, Complex host, int k, int ncomp)
{
    StencilOp op;
    op.dim = n;
    op.ncomp = ncomp;
    const int ks = host == Complex::Primal ? k : n - k;
    for (unsigned axes = 0; axes < (1u << n); ++axes) {
        if (std::popcount(axes) != ks) continue;
        const unsigned form_axes = host == Complex::Primal ? axes : stag::complement(axes, n);
        StencilRow row;
        row.dst_axes = axes;
        row.dst = stag::detail::stratum_index(axes, n);
        row.terms.push_back(
            StencilTerm{row.dst, {0, 0, 0}, Scalar(stag::complement_sign(form_axes, n))});
        op.rows.push_back(std::move(row));
    }
    return op;
}

} // namespace declib::detail
