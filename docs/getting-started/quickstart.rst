Quickstart
==========

Solve the Poisson problem :math:`\Delta u = f` for a 0-form on the unit
square.

C++
---

.. code-block:: cpp

   #include <declib/declib.hpp>

   int main(int argc, char** argv) {
       PetscInitialize(&argc, &argv, nullptr, nullptr);
       Kokkos::initialize(argc, argv);
       {
           declib::StructuredGrid<2> grid({128, 128}, {1.0, 1.0});
           declib::Cochain<0> u(grid);
           auto du = declib::d(u);              // a primal 1-cochain
           auto star_du = declib::hodge_star<1, 2>(du);
       }
       Kokkos::finalize();
       PetscFinalize();
   }

Python
------

.. code-block:: python

   import declib

   grid = declib.StructuredGrid((128, 128))
   u = declib.Cochain(grid, degree=0)
   du = declib.d(u)
   assert du.degree == 1

Next, read :doc:`/user-guide/concepts`.
