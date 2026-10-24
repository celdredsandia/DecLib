/**
 * @file grid_mapping.hpp
 * @brief Which complex hosts straight forms and which hosts twisted forms.
 */
#pragma once

#include "declib/core/types.hpp"
#include "declib/topology/cell_set.hpp"

namespace declib {

/**
 * @brief Assignment of straight and twisted forms to the primal and dual
 *        complexes.
 *
 * A \f$k\f$-form always lives on the \f$k\f$-cells of its host complex.
 * Primal cells carry inner orientations and dual cells outer orientations,
 * so a discretization must choose where each kind of form lives:
 *
 * - straight_on_primal(): the usual DEC convention. Potentials and
 *   circulations live on the primal grid, fluxes and densities on the dual.
 * - twisted_on_primal(): the finite-volume and TRiSK convention. Mass is a
 *   twisted \f$n\f$-form on primal cells and vorticity lives on dual
 *   vertices.
 *
 * Under either mapping a straight \f$k\f$-form and a twisted
 * \f$(n-k)\f$-form are stored on the same primal cells, so a diagonal Hodge
 * star is a pointwise scaling.
 */
class GridMapping {
public:
    /// @brief Straight forms on the primal complex, twisted forms on the dual.
    [[nodiscard]] static constexpr GridMapping straight_on_primal() noexcept
    {
        return GridMapping(Complex::Primal);
    }

    /// @brief Twisted forms on the primal complex, straight forms on the dual.
    [[nodiscard]] static constexpr GridMapping twisted_on_primal() noexcept
    {
        return GridMapping(Complex::Dual);
    }

    /// @brief Complex hosting forms of orientation @p o.
    [[nodiscard]] constexpr Complex host(Orientation o) const noexcept
    {
        return o == Orientation::Straight ? straight_host_ : opposite(straight_host_);
    }

    /// @brief Cells on which a \f$k\f$-form of orientation @p o is stored.
    [[nodiscard]] constexpr CellSet cells(int k, Orientation o) const noexcept
    {
        return CellSet{host(o), k};
    }

    /**
     * @brief The same cells, expressed as primal cells.
     *
     * Storage is always indexed by primal cells: a dual \f$k\f$-cell is
     * stored on the primal \f$(n-k)\f$-cell it crosses.
     *
     * @param k Form degree.
     * @param o Form orientation.
     * @param n Dimension of the complex.
     */
    [[nodiscard]] constexpr CellSet storage_cells(int k, Orientation o, int n) const noexcept
    {
        return host(o) == Complex::Primal ? CellSet{Complex::Primal, k}
                                          : CellSet{Complex::Primal, n - k};
    }

    /// @brief Equality of mappings.
    friend constexpr bool operator==(const GridMapping&, const GridMapping&) = default;

private:
    constexpr explicit GridMapping(Complex straight_host) noexcept : straight_host_(straight_host) {}

    Complex straight_host_;
};

} // namespace declib
