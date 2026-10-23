/**
 * @file declib.hpp
 * @brief Umbrella header for DecLib++.
 *
 * Including this header pulls in the full public C++ API: grids, cochains
 * (discrete differential forms) and the DEC operators acting on them.
 */
#pragma once

#include "declib/forms/cochain.hpp"
#include "declib/grid/structured_grid.hpp"
#include "declib/grid/unstructured_grid.hpp"
#include "declib/operators/exterior_derivative.hpp"
#include "declib/operators/hodge_star.hpp"

/**
 * @namespace declib
 * @brief Top-level namespace of DecLib++, a discrete exterior calculus
 *        library built on Kokkos and PETSc.
 */
namespace declib {

/// @brief Library version as a `major.minor.patch` string.
inline constexpr const char* version = "0.1.0";

} // namespace declib
