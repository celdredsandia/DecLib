// Quickstart: the Laplacian of a manufactured solution on the periodic unit
// square, built from d and the Hodge star, then a Poisson solve with KSP.
#include <cmath>
#include <numbers>

#include <declib/declib.hpp>

int main(int argc, char** argv)
{
    // PETSc initializes MPI and, on first use of a Kokkos vector, Kokkos.
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
    {
        using namespace declib;
        constexpr double pi = std::numbers::pi;

        auto topo = stag::Topology<2>::create(PETSC_COMM_WORLD, {.cells = {64, 64}});
        auto geom = stag::UniformGeometry<2>::create(topo, {.lengths = {1.0, 1.0}});
        auto M = DiscreteManifold::create(topo, geom);

        auto V0 = M->space(0, Orientation::Straight);  // primal vertices
        auto V1 = M->space(1, Orientation::Straight);  // primal edges
        auto tV1 = M->space(1, Orientation::Twisted);  // dual edges

        OpPtr d0 = ExteriorDerivative::create(V0);
        OpPtr s1 = HodgeStar::create(V1);
        OpPtr d1 = ExteriorDerivative::create(tV1);
        OpPtr s0 = HodgeStar::create(V0);
        OpPtr lap = inverse(s0) * d1 * s1 * d0; // spaces are checked here

        const auto u_exact = [&](std::span<const Real> x, std::span<Scalar> v) {
            v[0] = std::sin(2 * pi * x[0]) * std::sin(2 * pi * x[1]);
        };
        const Form u = V0->from_function(u_exact);
        Form f = (*lap)(u); // approximately -8 pi^2 u

        Form err = u.copy();
        err.scale(-8 * pi * pi).axpy(-1, f);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "max |lap u + 8 pi^2 u| / (8 pi^2) = %.3e\n",
                              static_cast<double>(err.norm_max() / (8 * pi * pi))));

        // Solve -lap u = g with the Hodge Laplacian plus a shift (the periodic
        // Laplacian alone has constants in its kernel).
        OpPtr A = hodge_laplacian(V0) + identity(V0);
        Form g = V0->from_function([&](std::span<const Real> x, std::span<Scalar> v) {
            v[0] = (8 * pi * pi + 1) * std::sin(2 * pi * x[0]) * std::sin(2 * pi * x[1]);
        });
        Form w = (*inverse(A, {.rtol = 1e-10}))(g);
        w.axpy(-1, u);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Poisson solve: max error %.3e\n", static_cast<double>(w.norm_max())));
    }
    PetscCall(PetscFinalize());
    return 0;
}
