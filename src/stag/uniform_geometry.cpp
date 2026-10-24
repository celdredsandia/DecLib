#include "declib/stag/uniform_geometry.hpp"

#include <bit>
#include <cmath>
#include <vector>

#include "declib/forms/form.hpp"
#include "declib/manifold/discrete_manifold.hpp"
#include "declib/stag/detail/layout.hpp"
#include "declib/stag/form_space.hpp"
#include "declib/stag/orientation.hpp"
#include "detail/petsc.hpp"
#include "stag/cell_field.hpp"

namespace declib::stag {

namespace {

// Gauss-Legendre nodes and weights on [0, 1].
void gauss_legendre(int npts, std::vector<double>& x, std::vector<double>& w)
{
    x.assign(npts, 0);
    w.assign(npts, 0);
    for (int i = 0; i < npts; ++i) {
        double z = std::cos(M_PI * (i + 0.75) / (npts + 0.5));
        double dp = 0;
        for (int it = 0; it < 100; ++it) {
            double p0 = 1, p1 = 0;
            for (int j = 1; j <= npts; ++j) {
                const double p2 = p1;
                p1 = p0;
                p0 = ((2.0 * j - 1) * z * p1 - (j - 1.0) * p2) / j;
            }
            dp = npts * (z * p0 - p1) / (z * z - 1);
            const double dz = p0 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-15) break;
        }
        x[i] = 0.5 * (1 - z);
        w[i] = 1.0 / ((1 - z * z) * dp * dp);
    }
}

// Position of the k-subset `axes` in the basis dx^I of k-forms, ordered
// lexicographically by sorted index tuple I. For n <= 3 that is the order of
// the bit masks.
int basis_index(unsigned axes, int n)
{
    const int k = std::popcount(axes);
    int idx = 0;
    for (unsigned m = 0; m < axes && m < (1u << n); ++m)
        if (std::popcount(m) == k) ++idx;
    return idx;
}

} // namespace

template <int Dim>
UniformGeometry<Dim>::UniformGeometry(std::shared_ptr<const Topology<Dim>> topology, const Options& options)
    : topology_(std::move(topology)), options_(options)
{
    for (int a = 0; a < Dim; ++a) {
        if (options_.lengths[a] == 0) options_.lengths[a] = 1;
        if (options_.lengths[a] < 0) throw Error("UniformGeometry: lengths must be positive");
    }
}

template <int Dim>
std::shared_ptr<const UniformGeometry<Dim>> UniformGeometry<Dim>::create(std::shared_ptr<const Topology<Dim>> topology,
                                                                         const Options& options)
{
    if (!topology) throw Error("UniformGeometry: null topology");
    return std::shared_ptr<const UniformGeometry<Dim>>(new UniformGeometry<Dim>(std::move(topology), options));
}

template <int Dim>
UniformGeometry<Dim>::~UniformGeometry() = default;

template <int Dim>
Real UniformGeometry<Dim>::spacing(int axis) const
{
    return options_.lengths.at(axis) / static_cast<Real>(topology_->options().cells.at(axis));
}

template <int Dim>
Real UniformGeometry<Dim>::length(int axis) const
{
    return options_.lengths.at(axis);
}

template <int Dim>
Real UniformGeometry<Dim>::origin(int axis) const
{
    return options_.origin.at(axis);
}

template <int Dim>
const declib::Topology& UniformGeometry<Dim>::topology() const noexcept
{
    return *topology_;
}

template <int Dim>
const CellField& UniformGeometry<Dim>::measure(CellSet cells) const
{
    std::lock_guard lock(mutex_);
    const auto key = std::make_pair(static_cast<int>(cells.complex), cells.k);
    if (auto it = measures_.find(key); it != measures_.end()) return *it->second;

    const int ks = cells.complex == Complex::Primal ? cells.k : Dim - cells.k;
    auto field = std::make_unique<detail::StagCellField>(topology_->create_compatible_dm(ks, 1), cells, 1);
    const auto layout = detail::local_layout(field->dm());
    const auto slots = detail::slots_of(field->dm(), Dim);
    const auto g = detail::local_layout(field->dm());
    PetscScalar* a;
    DECLIB_PETSC(VecGetArray(field->local_vec(), &a));
    for (const auto& s : topology_->strata(cells)) {
        const int slot = slots.s[detail::stratum_index(s.axes, Dim)][0];
        // Fill every ghosted point; values outside the domain are harmless.
        for (int k = g.g0[2]; k < g.g0[2] + g.gn[2]; ++k)
            for (int j = g.g0[1]; j < g.g0[1] + g.gn[1]; ++j)
                for (int i = g.g0[0]; i < g.g0[0] + g.gn[0]; ++i) {
                    const int idx[3] = {i, j, k};
                    Real m = 1;
                    for (int ax = 0; ax < Dim; ++ax) {
                        const bool in_primal = (s.axes >> ax) & 1u;
                        if (cells.complex == Complex::Primal) {
                            if (in_primal) m *= spacing(ax);
                        } else if (!in_primal) {
                            // Dual cell spans this axis; half cells on non-periodic boundary planes.
                            Real h = spacing(ax);
                            if (!topology_->periodic(ax) &&
                                (idx[ax] <= 0 || idx[ax] >= topology_->options().cells[ax]))
                                h *= 0.5;
                            m *= h;
                        }
                    }
                    a[layout.index(i, j, k, slot)] = m;
                }
    }
    DECLIB_PETSC(VecRestoreArray(field->local_vec(), &a));
    const CellField& ref = *field;
    measures_.emplace(key, std::move(field));
    return ref;
}

template <int Dim>
const CellField& UniformGeometry<Dim>::coordinates(Complex complex) const
{
    std::lock_guard lock(mutex_);
    const int key = static_cast<int>(complex);
    if (auto it = coordinates_.find(key); it != coordinates_.end()) return *it->second;
    // Primal vertices live on stratum 0; dual vertices on elements.
    const int ks = complex == Complex::Primal ? 0 : Dim;
    auto field = std::make_unique<detail::StagCellField>(topology_->create_compatible_dm(ks, Dim),
                                                         CellSet{complex, 0}, Dim);
    const auto layout = detail::local_layout(field->dm());
    const auto slots = detail::slots_of(field->dm(), Dim);
    const unsigned axes = complex == Complex::Primal ? 0u : (1u << Dim) - 1u;
    const int si = detail::stratum_index(axes, Dim);
    PetscScalar* a;
    DECLIB_PETSC(VecGetArray(field->local_vec(), &a));
    for (int k = layout.g0[2]; k < layout.g0[2] + layout.gn[2]; ++k)
        for (int j = layout.g0[1]; j < layout.g0[1] + layout.gn[1]; ++j)
            for (int i = layout.g0[0]; i < layout.g0[0] + layout.gn[0]; ++i) {
                const int idx[3] = {i, j, k};
                for (int ax = 0; ax < Dim; ++ax)
                    a[layout.index(i, j, k, slots.s[si][ax])] =
                        origin(ax) + (idx[ax] + (complex == Complex::Dual ? 0.5 : 0.0)) * spacing(ax);
            }
    DECLIB_PETSC(VecRestoreArray(field->local_vec(), &a));
    const CellField& ref = *field;
    coordinates_.emplace(key, std::move(field));
    return ref;
}

template <int Dim>
void UniformGeometry<Dim>::de_rham(const FormFunction& omega, Form& out, Quadrature quad) const
{
    const auto* space = dynamic_cast<const stag::FormSpace<Dim>*>(&out.space());
    if (!space) throw TypeError("UniformGeometry::de_rham: form is not on a structured grid of this dimension");
    const FormType& t = out.type();
    const Complex host = out.space().cells().complex;
    const int r = t.bundle.rank();
    int nbasis = 1;
    for (int i = 0; i < t.degree; ++i) nbasis = nbasis * (Dim - i) / (i + 1);

    const int npts = std::max(1, (quad.order + 2) / 2);
    std::vector<double> qx, qw;
    gauss_legendre(npts, qx, qw);

    DM dm = space->dm();
    const auto layout = detail::local_layout(dm);
    const auto slots = detail::slots_of(dm, Dim);
    Vec loc;
    DECLIB_PETSC(DMGetLocalVector(dm, &loc));
    DECLIB_PETSC(VecSet(loc, 0));
    PetscScalar* a;
    DECLIB_PETSC(VecGetArray(loc, &a));
    std::vector<Real> x(Dim);
    std::vector<Scalar> vals(static_cast<std::size_t>(nbasis * r));

    for (const auto& s : space->strata()) {
        const int si = detail::stratum_index(s.axes, Dim);
        // Axes along which the integration cell extends.
        const unsigned cell_axes = host == Complex::Primal ? s.axes : complement(s.axes, Dim);
        const int b = basis_index(cell_axes, Dim);
        const int kc = std::popcount(cell_axes);
        int nq = 1;
        for (int i = 0; i < kc; ++i) nq *= npts;
        const auto box = detail::owned_cells(dm, s.axes);
        for (int k = box.lo[2]; k < box.hi[2]; ++k)
            for (int j = box.lo[1]; j < box.hi[1]; ++j)
                for (int i = box.lo[0]; i < box.hi[0]; ++i) {
                    const int idx[3] = {i, j, k};
                    // Cell extent per axis: [lo, hi] (lo == hi on axes it does not span).
                    double lo[3], hi[3];
                    for (int ax = 0; ax < Dim; ++ax) {
                        const double h = spacing(ax);
                        const double xi = origin(ax) + idx[ax] * h;
                        if (!((cell_axes >> ax) & 1u)) {
                            lo[ax] = hi[ax] = host == Complex::Primal ? xi : xi + 0.5 * h;
                        } else if (host == Complex::Primal) {
                            lo[ax] = xi;
                            hi[ax] = xi + h;
                        } else {
                            lo[ax] = xi - 0.5 * h;
                            hi[ax] = xi + 0.5 * h;
                            if (!topology_->periodic(ax)) {
                                lo[ax] = std::max(lo[ax], origin(ax));
                                hi[ax] = std::min(hi[ax], origin(ax) + length(ax));
                            }
                        }
                    }
                    std::vector<Scalar> acc(static_cast<std::size_t>(r), 0);
                    for (int q = 0; q < nq; ++q) {
                        double wgt = 1;
                        int rem = q;
                        for (int ax = 0; ax < Dim; ++ax) {
                            if ((cell_axes >> ax) & 1u) {
                                const int p = rem % npts;
                                rem /= npts;
                                x[ax] = lo[ax] + qx[p] * (hi[ax] - lo[ax]);
                                wgt *= qw[p] * (hi[ax] - lo[ax]);
                            } else {
                                x[ax] = lo[ax];
                            }
                        }
                        std::fill(vals.begin(), vals.end(), 0);
                        omega(std::span<const Real>(x), std::span<Scalar>(vals));
                        for (int c = 0; c < r; ++c) acc[c] += wgt * vals[b * r + c];
                    }
                    for (int c = 0; c < r; ++c) a[layout.index(i, j, k, slots.s[si][c])] = acc[c];
                }
    }
    DECLIB_PETSC(VecRestoreArray(loc, &a));
    DECLIB_PETSC(DMLocalToGlobal(dm, loc, INSERT_VALUES, out.global_vec()));
    DECLIB_PETSC(DMRestoreLocalVector(dm, &loc));
}

template class UniformGeometry<1>;
template class UniformGeometry<2>;
template class UniformGeometry<3>;

} // namespace declib::stag
