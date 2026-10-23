/**
 * @file declib.hpp
 * @brief Umbrella header for DecLib++.
 *
 * Including this header pulls in the full public C++ API: topology, grid
 * mapping, geometry, discrete manifolds, forms, operators and the
 * structured (`DMStag`) backend.
 */
#pragma once

#include "declib/core/error.hpp"
#include "declib/core/types.hpp"
#include "declib/forms/boundary_condition.hpp"
#include "declib/forms/bundle.hpp"
#include "declib/forms/form.hpp"
#include "declib/forms/form_space.hpp"
#include "declib/forms/form_type.hpp"
#include "declib/geometry/cell_field.hpp"
#include "declib/geometry/connection.hpp"
#include "declib/geometry/geometry.hpp"
#include "declib/manifold/discrete_manifold.hpp"
#include "declib/mapping/grid_mapping.hpp"
#include "declib/operators/bilinear_operator.hpp"
#include "declib/operators/derived.hpp"
#include "declib/operators/exterior_derivative.hpp"
#include "declib/operators/hodge_star.hpp"
#include "declib/operators/interior_product.hpp"
#include "declib/operators/linear_operator.hpp"
#include "declib/operators/nonlinear_operator.hpp"
#include "declib/operators/pairing.hpp"
#include "declib/operators/wedge.hpp"
#include "declib/stag/form_space.hpp"
#include "declib/stag/form_view.hpp"
#include "declib/stag/index_box.hpp"
#include "declib/stag/orientation.hpp"
#include "declib/stag/topology.hpp"
#include "declib/stag/uniform_geometry.hpp"
#include "declib/topology/cell_set.hpp"
#include "declib/topology/topology.hpp"

/**
 * @namespace declib
 * @brief Top-level namespace of DecLib++, a discrete exterior calculus
 *        library built on Kokkos and PETSc.
 */
namespace declib {

/// @brief Library version as a `major.minor.patch` string.
inline constexpr const char* version = "0.1.0";

} // namespace declib

/**
 * @namespace declib::stag
 * @brief Structured backend: topologically Cartesian grids on PETSc `DMStag`.
 */
