#include "declib/stag/topology.hpp"

#include <algorithm>
#include <array>
#include <bit>

#include "declib/stag/detail/layout.hpp"
#include "detail/petsc.hpp"
#include "stag/stencil.hpp"

namespace declib::stag {

namespace {

DMBoundaryType to_petsc(Boundary b)
{
    return b == Boundary::Periodic ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_GHOSTED;
}

template <int Dim>
constexpr std::array<Stratum, (1u << Dim)> make_strata()
{
    std::array<Stratum, (1u << Dim)> s{};
    int idx = 0;
    for (int k = 0; k <= Dim; ++k)
        for (unsigned axes = 0; axes < (1u << Dim); ++axes)
            if (std::popcount(axes) == k) s[idx++] = Stratum{DMSTAG_NULL_LOCATION, k, axes};
    return s;
}

template <int Dim>
const std::array<Stratum, (1u << Dim)>& strata_table()
{
    static const auto table = [] {
        auto s = make_strata<Dim>();
        for (auto& st : s) st.location = detail::location_of(st.axes, Dim);
        return s;
    }();
    return table;
}

// Primal dimension of the strata storing a cell set.
int storage_dim(CellSet c, int n) { return c.complex == Complex::Primal ? c.k : n - c.k; }

} // namespace

template <int Dim>
Topology<Dim>::Topology(MPI_Comm comm, const Options& options) : options_(options)
{
    for (int a = 0; a < Dim; ++a) {
        if (options_.cells[a] < 1) throw Error("stag::Topology: every axis needs at least one cell");
        if (options_.boundary[a] == Boundary::Periodic && options_.cells[a] < options_.stencil_width)
            throw Error("stag::Topology: a periodic axis needs at least stencil_width cells");
    }
    if (options_.stencil_width < 1) throw Error("stag::Topology: stencil_width must be at least 1");
    const auto& c = options_.cells;
    const auto& b = options_.boundary;
    const auto r = [&](int a) { return options_.ranks[a] > 0 ? options_.ranks[a] : PETSC_DECIDE; };
    const PetscInt sw = options_.stencil_width;
    if constexpr (Dim == 1) {
        DECLIB_PETSC(DMStagCreate1d(comm, to_petsc(b[0]), c[0], 1, 1, DMSTAG_STENCIL_BOX, sw, nullptr, &dm_));
    } else if constexpr (Dim == 2) {
        DECLIB_PETSC(DMStagCreate2d(comm, to_petsc(b[0]), to_petsc(b[1]), c[0], c[1], r(0), r(1), 1, 1, 1,
                                    DMSTAG_STENCIL_BOX, sw, nullptr, nullptr, &dm_));
    } else {
        DECLIB_PETSC(DMStagCreate3d(comm, to_petsc(b[0]), to_petsc(b[1]), to_petsc(b[2]), c[0], c[1], c[2], r(0),
                                    r(1), r(2), 1, 1, 1, 1, DMSTAG_STENCIL_BOX, sw, nullptr, nullptr, nullptr,
                                    &dm_));
    }
    DECLIB_PETSC(DMSetVecType(dm_, VECKOKKOS));
    DECLIB_PETSC(DMSetMatType(dm_, MATAIJKOKKOS));
    DECLIB_PETSC(DMSetUp(dm_));
}

template <int Dim>
std::shared_ptr<const Topology<Dim>> Topology<Dim>::create(MPI_Comm comm, const Options& options)
{
    return std::shared_ptr<const Topology<Dim>>(new Topology<Dim>(comm, options));
}

template <int Dim>
Topology<Dim>::~Topology()
{
    DMDestroy(&dm_);
}

template <int Dim>
DM Topology<Dim>::base_dm() const noexcept
{
    return dm_;
}

template <int Dim>
IndexBox<Dim> Topology<Dim>::owned_box() const
{
    PetscInt x[3], m[3];
    DECLIB_PETSC(DMStagGetCorners(dm_, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], nullptr, nullptr, nullptr));
    IndexBox<Dim> box;
    for (int a = 0; a < Dim; ++a) {
        box.start[a] = x[a];
        box.extent[a] = m[a];
    }
    return box;
}

template <int Dim>
IndexBox<Dim> Topology<Dim>::ghosted_box() const
{
    PetscInt x[3], m[3];
    DECLIB_PETSC(DMStagGetGhostCorners(dm_, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2]));
    IndexBox<Dim> box;
    for (int a = 0; a < Dim; ++a) {
        box.start[a] = x[a];
        box.extent[a] = m[a];
    }
    return box;
}

template <int Dim>
const typename Topology<Dim>::Options& Topology<Dim>::options() const noexcept
{
    return options_;
}

template <int Dim>
bool Topology<Dim>::periodic(int axis) const
{
    if (axis < 0 || axis >= Dim) throw Error("stag::Topology::periodic: axis out of range");
    return options_.boundary[axis] == Boundary::Periodic;
}

template <int Dim>
std::span<const Stratum> Topology<Dim>::all_strata() noexcept
{
    return strata_table<Dim>();
}

template <int Dim>
std::span<const Stratum> Topology<Dim>::strata(CellSet cells) const
{
    if (cells.k < 0 || cells.k > Dim) throw Error("stag::Topology::strata: cell dimension out of range");
    const int k = storage_dim(cells, Dim);
    const auto& t = strata_table<Dim>();
    const auto first = std::find_if(t.begin(), t.end(), [&](const Stratum& s) { return s.k == k; });
    const auto last = std::find_if(first, t.end(), [&](const Stratum& s) { return s.k != k; });
    return {first, last};
}

template <int Dim>
DM Topology<Dim>::create_compatible_dm(int k, int dof) const
{
    PetscInt d[4] = {0, 0, 0, 0};
    d[k] = dof;
    DM out;
    DECLIB_PETSC(DMStagCreateCompatibleDMStag(dm_, d[0], d[1], d[2], d[3], &out));
    DECLIB_PETSC(DMSetVecType(out, VECKOKKOS));
    DECLIB_PETSC(DMSetMatType(out, MATAIJKOKKOS));
    return out;
}

template <int Dim>
MPI_Comm Topology<Dim>::comm() const noexcept
{
    return PetscObjectComm(reinterpret_cast<PetscObject>(dm_));
}

template <int Dim>
std::size_t Topology<Dim>::num_owned(CellSet cells) const
{
    std::size_t n = 0;
    for (const auto& s : strata(cells)) n += static_cast<std::size_t>(detail::owned_cells(dm_, s.axes).size());
    return n;
}

template <int Dim>
std::size_t Topology<Dim>::num_local(CellSet cells) const
{
    const auto g = ghosted_box();
    std::size_t per = 1;
    for (int a = 0; a < Dim; ++a) per *= static_cast<std::size_t>(g.extent[a]);
    return per * strata(cells).size();
}

template <int Dim>
PetscInt Topology<Dim>::num_global(CellSet cells) const
{
    PetscInt n = 0;
    for (const auto& s : strata(cells)) {
        PetscInt c = 1;
        for (int a = 0; a < Dim; ++a) {
            const bool extends = (s.axes >> a) & 1u;
            c *= options_.cells[a] + ((extends || periodic(a)) ? 0 : 1);
        }
        n += c;
    }
    return n;
}

template <int Dim>
Mat Topology<Dim>::incidence(CellSet cells) const
{
    if (cells.k < 1 || cells.k > Dim) throw Error("stag::Topology::incidence: need 1 <= k <= n");
    // The boundary operator is the transpose of the exterior derivative on
    // (k-1)-cochains of the same complex.
    const int kd = storage_dim(CellSet{cells.complex, cells.k - 1}, Dim);
    const int kr = storage_dim(cells, Dim);
    DM xdm = create_compatible_dm(kd, 1);
    DM ydm = create_compatible_dm(kr, 1);
    const auto op = declib::detail::exterior_derivative_stencil(Dim, cells.complex, cells.k - 1, 1);
    Mat d = op.assemble(xdm, ydm, MATAIJ);
    Mat dt;
    DECLIB_PETSC(MatTranspose(d, MAT_INITIAL_MATRIX, &dt));
    DECLIB_PETSC(MatDestroy(&d));
    DECLIB_PETSC(DMDestroy(&xdm));
    DECLIB_PETSC(DMDestroy(&ydm));
    return dt;
}

template <int Dim>
DualBoundary Topology<Dim>::dual_boundary() const noexcept
{
    return options_.dual_boundary;
}

template <int Dim>
int Topology<Dim>::stencil_width() const noexcept
{
    return options_.stencil_width;
}

template class Topology<1>;
template class Topology<2>;
template class Topology<3>;

} // namespace declib::stag
