/**
 * @file types.hpp
 * @brief Basic types and enumerations shared by every DecLib++ component.
 */
#pragma once

#include <Kokkos_Core.hpp>
#include <petscsystypes.h>

#if defined(PETSC_USE_COMPLEX)
#error "DecLib++ requires PETSc configured with real scalars (--with-scalar-type=real)"
#endif

namespace declib {

/// @brief Value type of every form, operator and functional. Always real.
using Scalar = PetscScalar;

/// @brief Real type used for geometric quantities (lengths, areas, volumes).
using Real = PetscReal;

/// @brief Kokkos execution space used by all kernels.
using ExecSpace = Kokkos::DefaultExecutionSpace;

/// @brief Kokkos memory space matching ExecSpace.
using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;

/**
 * @brief Orientation type of a differential form.
 *
 * A straight (inner oriented) \f$k\f$-form is integrated over \f$k\f$-cells
 * that carry an inner orientation. A twisted (outer oriented) \f$k\f$-form,
 * also called a pseudo-form or density, is integrated over cells that carry
 * an outer orientation. The Hodge star maps one kind to the other.
 */
enum class Orientation {
    Straight, ///< Inner oriented form.
    Twisted   ///< Outer oriented form.
};

/// @brief The other orientation: straight for twisted and vice versa.
constexpr Orientation opposite(Orientation o) noexcept
{
    return o == Orientation::Straight ? Orientation::Twisted : Orientation::Straight;
}

/**
 * @brief Orientation of a wedge product of two forms.
 *
 * Straight \f$\wedge\f$ straight and twisted \f$\wedge\f$ twisted are
 * straight; a mixed product is twisted.
 */
constexpr Orientation product_orientation(Orientation a, Orientation b) noexcept
{
    return a == b ? Orientation::Straight : Orientation::Twisted;
}

/**
 * @brief The primal cell complex or its dual.
 *
 * Primal cells carry inner orientations. A dual \f$k\f$-cell crosses one
 * primal \f$(n-k)\f$-cell and inherits an outer orientation from it.
 */
enum class Complex {
    Primal, ///< The mesh itself.
    Dual    ///< The dual complex, one dual \f$k\f$-cell per primal \f$(n-k)\f$-cell.
};

/// @brief The other complex: dual for primal and vice versa.
constexpr Complex opposite(Complex c) noexcept
{
    return c == Complex::Primal ? Complex::Dual : Complex::Primal;
}

/// @brief Mesh backend implementing a Topology.
enum class Backend {
    Stag, ///< Topologically structured grids on PETSc `DMStag`.
    Plex  ///< Unstructured meshes on PETSc `DMPlex` (not yet implemented).
};

} // namespace declib
