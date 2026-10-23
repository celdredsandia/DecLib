/**
 * @file uniform_geometry.hpp
 * @brief Uniformly spaced Cartesian geometry on a structured topology.
 */
#pragma once

#include <array>
#include <memory>

#include "declib/core/types.hpp"
#include "declib/geometry/geometry.hpp"
#include "declib/stag/topology.hpp"

namespace declib::stag {

/**
 * @brief Constant spacing \f$h_i = L_i / N_i\f$ along each axis.
 *
 * Orthogonal, with circumcentric (equivalently barycentric) dual vertices
 * at element centres. Measures are products of spacings and are computed
 * on the fly inside kernels; measure() materialises them only on request.
 *
 * Rectilinear (tensor-product spacing) and mapped (curvilinear) geometries
 * follow in phase 3 with the same interface.
 *
 * @tparam Dim Grid dimension (1, 2 or 3).
 */
template <int Dim>
class UniformGeometry final : public declib::Geometry {
public:
    /// @brief Creation options.
    struct Options {
        std::array<Real, Dim> lengths{}; ///< Domain extent \f$L_i\f$ per axis.
        std::array<Real, Dim> origin{};  ///< Coordinates of the low corner.
    };

    /// @brief Attach a uniform geometry to @p topology.
    [[nodiscard]] static std::shared_ptr<const UniformGeometry>
    create(std::shared_ptr<const Topology<Dim>> topology, const Options& options);

    /// @brief Spacing \f$h_i\f$ along @p axis.
    [[nodiscard]] Real spacing(int axis) const;

    [[nodiscard]] const declib::Topology& topology() const noexcept override;
    [[nodiscard]] DualPlacement dual_placement() const noexcept override { return DualPlacement::Circumcentric; }
    [[nodiscard]] const CellField& measure(CellSet cells) const override;
    [[nodiscard]] const CellField& coordinates(Complex complex) const override;
    [[nodiscard]] bool is_orthogonal() const noexcept override { return true; }
    void de_rham(const FormFunction& omega, Form& out, Quadrature quad = {}) const override;

private:
    UniformGeometry(std::shared_ptr<const Topology<Dim>> topology, const Options& options);

    std::shared_ptr<const Topology<Dim>> topology_;
    Options options_;
};

} // namespace declib::stag
