Quickstart
==========

Build the Laplacian :math:`\nabla^2 u = \star^{-1} d \star d\, u` of a 0-form
on the periodic unit square from the exterior derivative and the Hodge star,
apply it to a manufactured solution, and solve a shifted Poisson problem.
Both programs live in ``examples/`` and run as part of the test suite.

C++
---

.. literalinclude:: ../../examples/quickstart.cpp
   :language: cpp
   :lines: 4-

Run it with ``./build/examples/quickstart``. It prints the relative error of
the discrete Laplacian, about :math:`8 \times 10^{-4}` on the
:math:`64 \times 64` grid and four times smaller on each refinement.

PETSc initializes MPI, and Kokkos on first use of a Kokkos vector, so a
program needs only ``PetscInitialize`` and ``PetscFinalize``. Every DecLib++
object must be destroyed before ``PetscFinalize``; the inner scope above takes
care of that.

Python
------

.. literalinclude:: ../../examples/quickstart.py
   :language: python
   :lines: 3-

Spaces and operators are the same objects as in C++; operators compose with
``@`` and :meth:`~declib.operators.LinearOperator.assemble` returns a
``petsc4py.PETSc.Mat`` ready for ``KSP``, ``SNES`` or ``TS``.

Next, read :doc:`/user-guide/concepts`.
