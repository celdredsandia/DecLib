#include "declib/stag/form_space.hpp"

#include <vector>

#include "declib/manifold/discrete_manifold.hpp"
#include "declib/stag/detail/layout.hpp"
#include "detail/petsc.hpp"

namespace declib::stag {

template <int Dim>
FormSpace<Dim>::FormSpace(std::shared_ptr<const DiscreteManifold> manifold, FormType type)
    : declib::FormSpace(manifold, type),
      topology_(dynamic_cast<const Topology<Dim>*>(&manifold->topology()))
{
    if (!topology_) throw TypeError("stag::FormSpace: topology is not a structured grid of this dimension");
    if (type.degree < 0 || type.degree > Dim) throw TypeError("stag::FormSpace: degree out of range");
    dm_ = topology_->create_compatible_dm(storage_dimension(), type.bundle.rank());
}

template <int Dim>
FormSpace<Dim>::FormSpace(const FormSpace& other) : declib::FormSpace(other), topology_(other.topology_), dm_(other.dm_)
{
    DECLIB_PETSC(PetscObjectReference(reinterpret_cast<PetscObject>(dm_)));
}

template <int Dim>
FormSpace<Dim>::~FormSpace()
{
    DMDestroy(&dm_);
}

template <int Dim>
const Topology<Dim>& FormSpace<Dim>::topology() const noexcept
{
    return *topology_;
}

template <int Dim>
int FormSpace<Dim>::storage_dimension() const noexcept
{
    return cells().complex == Complex::Primal ? type().degree : Dim - type().degree;
}

template <int Dim>
std::span<const Stratum> FormSpace<Dim>::strata() const
{
    return topology_->strata(cells());
}

template <int Dim>
DM FormSpace<Dim>::dm() const noexcept
{
    return dm_;
}

template <int Dim>
std::size_t FormSpace<Dim>::local_size() const
{
    return topology_->num_owned(cells()) * static_cast<std::size_t>(type().bundle.rank());
}

template <int Dim>
PetscInt FormSpace<Dim>::global_size() const
{
    return topology_->num_global(cells()) * type().bundle.rank();
}

template <int Dim>
void FormSpace<Dim>::validate_region(const BoundaryRegion& region) const
{
    if (!region.label_name().empty()) throw TypeError("structured grids have no labelled boundary regions");
    if (cells().complex != Complex::Primal)
        throw TypeError("essential conditions need forms hosted on the primal complex; forms on the dual "
                        "complex take boundary data through the extended dual cells");
    if (region.axis()) {
        const int a = *region.axis();
        if (a < 0 || a >= Dim) throw TypeError("boundary region axis out of range");
        if (topology_->periodic(a)) throw TypeError("boundary region names a periodic axis");
    } else {
        bool any = false;
        for (int a = 0; a < Dim; ++a) any = any || !topology_->periodic(a);
        if (!any) throw TypeError("the grid is periodic in every direction and has no boundary");
    }
}

template <int Dim>
IS FormSpace<Dim>::boundary_indices(const BoundaryRegion& region) const
{
    validate_region(region);
    const auto layout = detail::local_layout(dm_);
    const auto slots = detail::slots_of(dm_, Dim);
    const auto& cells_per_axis = topology_->options().cells;
    std::vector<PetscInt> local;
    for (const auto& s : strata()) {
        const int si = detail::stratum_index(s.axes, Dim);
        const auto box = detail::owned_cells(dm_, s.axes);
        for (int k = box.lo[2]; k < box.hi[2]; ++k)
            for (int j = box.lo[1]; j < box.hi[1]; ++j)
                for (int i = box.lo[0]; i < box.hi[0]; ++i) {
                    const int idx[3] = {i, j, k};
                    // A cell lies in a boundary plane of axis a if it does not
                    // extend along a and sits at index 0 or N.
                    const auto on_face = [&](int a, Side side) {
                        if (topology_->periodic(a) || ((s.axes >> a) & 1u)) return false;
                        return side == Side::Low ? idx[a] == 0 : idx[a] == cells_per_axis[a];
                    };
                    bool hit = false;
                    if (region.axis()) {
                        hit = on_face(*region.axis(), region.side());
                    } else {
                        for (int a = 0; a < Dim && !hit; ++a) hit = on_face(a, Side::Low) || on_face(a, Side::High);
                    }
                    if (!hit) continue;
                    for (int c = 0; c < type().bundle.rank(); ++c) local.push_back(layout.index(i, j, k, slots.s[si][c]));
                }
    }
    ISLocalToGlobalMapping l2g;
    DECLIB_PETSC(DMGetLocalToGlobalMapping(dm_, &l2g));
    DECLIB_PETSC(ISLocalToGlobalMappingApply(l2g, static_cast<PetscInt>(local.size()), local.data(), local.data()));
    IS is;
    DECLIB_PETSC(ISCreateGeneral(PetscObjectComm(reinterpret_cast<PetscObject>(dm_)),
                                 static_cast<PetscInt>(local.size()), local.data(), PETSC_COPY_VALUES, &is));
    return is;
}

template <int Dim>
SpacePtr FormSpace<Dim>::clone_with(std::vector<EssentialCondition> conditions) const
{
    auto copy = std::shared_ptr<FormSpace<Dim>>(new FormSpace<Dim>(*this));
    copy->essential_ = std::move(conditions);
    return copy;
}

template class FormSpace<1>;
template class FormSpace<2>;
template class FormSpace<3>;

} // namespace declib::stag
