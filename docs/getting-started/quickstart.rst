Quickstart
==========

Build the Hodge Laplacian :math:`\Delta = \star^{-1} d \star d` on 0-forms on
the periodic unit square and apply it to a manufactured solution.

.. note::

   The interfaces are in place; implementations land in phase 1 of the
   :doc:`/developer/architecture` roadmap.

C++
---

.. code-block:: cpp

   #include <declib/declib.hpp>

   int main(int argc, char** argv) {
       PetscInitialize(&argc, &argv, nullptr, nullptr);
       Kokkos::initialize(argc, argv);
       {
           using namespace declib;
           auto topo = stag::Topology<2>::create(PETSC_COMM_WORLD, {.cells = {128, 128}});
           auto geom = stag::UniformGeometry<2>::create(topo, {.lengths = {1.0, 1.0}});
           auto M    = DiscreteManifold::create(topo, geom);

           auto V0  = M->space(0, Orientation::Straight);  // primal vertices
           auto V1  = M->space(1, Orientation::Straight);  // primal edges
           auto tV1 = M->space(1, Orientation::Twisted);   // dual edges

           OpPtr d0 = ExteriorDerivative::create(V0);
           OpPtr s1 = HodgeStar::create(V1);
           OpPtr d1 = ExteriorDerivative::create(tV1);
           OpPtr s0 = HodgeStar::create(V0);
           OpPtr lap = inverse(s0) * d1 * s1 * d0;        // spaces checked here

           Form u = V0->from_function([](auto x, auto v) {
               v[0] = std::sin(2 * M_PI * x[0]) * std::sin(2 * M_PI * x[1]);
           });
           Form f = (*lap)(u);
       }
       Kokkos::finalize();
       PetscFinalize();
   }

Python
------

.. code-block:: python

   import numpy as np
   import declib as dl

   topo = dl.StagTopology((128, 128))
   geom = dl.UniformGeometry(topo, lengths=(1.0, 1.0))
   M = dl.DiscreteManifold(topo, geom)

   S, T = dl.Orientation.STRAIGHT, dl.Orientation.TWISTED
   V0, V1, tV1 = M.space(0, S), M.space(1, S), M.space(1, T)

   d0, d1 = dl.ExteriorDerivative(V0), dl.ExteriorDerivative(tV1)
   s0, s1 = dl.HodgeStar(V0), dl.HodgeStar(V1)
   lap = dl.inverse(s0) @ d1 @ s1 @ d0

   u = V0.from_function(lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))
   f = lap(u)
   A = lap.assemble()     # petsc4py.PETSc.Mat

Next, read :doc:`/user-guide/concepts`.
