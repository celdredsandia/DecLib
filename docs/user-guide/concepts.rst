Core concepts
=============

Cell complexes
--------------

A :cpp:class:`declib::Grid` is a cell complex of dimension :math:`n`. Its
:math:`k`-cells are vertices (:math:`k=0`), edges (:math:`k=1`), faces
(:math:`k=2`) and volumes (:math:`k=3`). Each cell carries an orientation.

Cochains
--------

A :math:`k`-cochain (:cpp:class:`declib::Cochain`, :py:class:`declib.forms.Cochain`)
assigns a number to every oriented :math:`k`-cell. It is the discrete form of
a differential :math:`k`-form :math:`\omega`, through the de Rham map

.. math::

   R(\omega)_\sigma = \int_\sigma \omega .

Operators
---------

Exterior derivative
   :math:`d_k = \partial_{k+1}^{T}`, so the discrete Stokes theorem is exact
   and :math:`d \circ d = 0`. See :cpp:func:`declib::d`.

Hodge star
   Maps primal :math:`k`-cochains to dual :math:`(n-k)`-cochains and carries
   all the metric information. See :cpp:func:`declib::hodge_star`.

Codifferential and Laplacian
   :math:`\delta = (-1)^{n(k+1)+1} \star d \star` and
   :math:`\Delta = d\delta + \delta d`. See :py:func:`declib.operators.laplacian`.
