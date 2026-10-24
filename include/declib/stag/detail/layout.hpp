// Internal: index arithmetic on ghosted DMStag local vectors.
#pragma once

#include <array>

#include <petscdmstag.h>

#include "declib/core/types.hpp"

namespace declib::stag::detail {

/// Maximum number of strata (2^3) and fiber components handled by kernels.
inline constexpr int max_strata = 8;

// Ghosted local layout of a DMStag, always padded to three axes. Element
// indices are global; entries for each point are contiguous.
struct LocalLayout {
    int g0[3] = {0, 0, 0}; // first ghosted element index per axis
    int gn[3] = {1, 1, 1}; // ghosted extent per axis
    int epe = 0;           // entries per element (point)

    KOKKOS_INLINE_FUNCTION int index(int i, int j, int k, int slot) const
    {
        return (((k - g0[2]) * gn[1] + (j - g0[1])) * gn[0] + (i - g0[0])) * epe + slot;
    }
};

// Half-open box of element indices, padded to three axes.
struct Box {
    int lo[3] = {0, 0, 0};
    int hi[3] = {1, 1, 1};

    [[nodiscard]] long size() const
    {
        long s = 1;
        for (int a = 0; a < 3; ++a) s *= (hi[a] > lo[a] ? hi[a] - lo[a] : 0);
        return s;
    }
};

// Layout of the ghosted local vector of `dm`.
LocalLayout local_layout(DM dm);

// Elements owned by this rank at which cells of the stratum with axis mask
// `axes` exist: one extra plane on the high boundary of non-periodic axes
// for strata that do not extend along that axis.
Box owned_cells(DM dm, unsigned axes);

// DMStag location of the stratum with axis mask `axes` in dimension `dim`.
DMStagStencilLocation location_of(unsigned axes, int dim);

// Index of the stratum with axis mask `axes` in Topology::all_strata().
int stratum_index(unsigned axes, int dim);

// Slots of every stratum and component in `dm`: slots[s][c], or -1 where the
// stratum carries no dofs. `ncomp` is at most `max_components`.
inline constexpr int max_components = 9;
struct Slots {
    int s[max_strata][max_components];
};
Slots slots_of(DM dm, int dim);

} // namespace declib::stag::detail
