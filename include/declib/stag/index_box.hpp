/**
 * @file index_box.hpp
 * @brief Rectangular ranges of element indices on a structured grid.
 */
#pragma once

#include <array>

#include <petscsystypes.h>

namespace declib::stag {

/**
 * @brief A box of element indices \f$[\mathrm{start}, \mathrm{start} + \mathrm{extent})\f$ per axis.
 * @tparam Dim Grid dimension (1, 2 or 3).
 */
template <int Dim>
struct IndexBox {
    std::array<PetscInt, Dim> start{};  ///< First index along each axis.
    std::array<PetscInt, Dim> extent{}; ///< Number of indices along each axis.
};

} // namespace declib::stag
