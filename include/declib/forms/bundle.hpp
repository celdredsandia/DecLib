/**
 * @file bundle.hpp
 * @brief Fibers of vector-bundle valued forms.
 */
#pragma once

#include "declib/core/types.hpp"

namespace declib {

/// @brief Kind of vector bundle a form takes values in.
enum class BundleKind {
    Scalar,    ///< Ordinary (real-valued) forms, rank 1.
    Trivial,   ///< \f$M \times \mathbb{R}^r\f$ with a global frame.
    Tangent,   ///< Tangent bundle, rank \f$n\f$ (vector-valued forms).
    Cotangent, ///< Cotangent bundle, rank \f$n\f$ (covector-valued forms, e.g. momentum).
    Custom     ///< Any other bundle; needs a user-supplied Connection.
};

/**
 * @brief The vector bundle \f$E\f$ in which a form takes values.
 *
 * An \f$E\f$-valued \f$k\f$-form stores one fiber element (\f$r\f$ numbers)
 * per \f$k\f$-cell, expressed in a frame attached to that cell. For a
 * trivial bundle the frame is global and DEC operators act componentwise.
 * Otherwise operators that compare fibers at different cells go through a
 * Connection.
 *
 * The fiber orientation lets a FormType describe objects such as a
 * covector-valued twisted \f$n\f$-form, or a twisted form valued in
 * straight 1-forms.
 */
class Bundle {
public:
    /// @brief Real-valued forms.
    [[nodiscard]] static Bundle scalar() noexcept;

    /// @brief \f$\mathbb{R}^r\f$-valued forms with a global frame.
    [[nodiscard]] static Bundle trivial(int rank);

    /// @brief Tangent-vector valued forms on an \f$n\f$-manifold.
    [[nodiscard]] static Bundle tangent(int n);

    /// @brief Covector valued forms on an \f$n\f$-manifold, with the given fiber orientation.
    [[nodiscard]] static Bundle cotangent(int n, Orientation fiber_orientation);

    /// @brief Kind of bundle.
    [[nodiscard]] BundleKind kind() const noexcept;

    /// @brief Fiber dimension \f$r\f$.
    [[nodiscard]] int rank() const noexcept;

    /// @brief Orientation of the fiber elements (straight unless stated).
    [[nodiscard]] Orientation fiber_orientation() const noexcept;

    /// @brief True for scalar and trivial bundles.
    [[nodiscard]] bool is_trivial() const noexcept;

    /// @brief The dual bundle \f$E^*\f$, used by FormSpace::dual_space().
    [[nodiscard]] Bundle dual() const;

    /// @brief Equality of kind, rank and fiber orientation.
    friend bool operator==(const Bundle&, const Bundle&) = default;

private:
    Bundle(BundleKind kind, int rank, Orientation fiber_orientation) noexcept;

    BundleKind kind_;
    int rank_;
    Orientation fiber_orientation_;
};

} // namespace declib
