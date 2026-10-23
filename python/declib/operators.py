"""DEC operators: exterior derivative, Hodge star and Laplacian."""

from __future__ import annotations

from typing import Literal

from declib import _core
from declib.forms import Cochain


def d(omega: Cochain) -> Cochain:
    """Discrete exterior derivative.

    Maps a :math:`k`-cochain to a :math:`(k+1)`-cochain such that
    :math:`d \\circ d = 0` exactly. See :cpp:func:`declib::d`.

    Parameters
    ----------
    omega : Cochain
        Input :math:`k`-cochain.

    Returns
    -------
    Cochain
        The :math:`(k+1)`-cochain :math:`d\\omega`.
    """
    return _core.d(omega)


def hodge_star(omega: Cochain,
               kind: Literal["diagonal", "galerkin"] = "diagonal") -> Cochain:
    """Discrete Hodge star :math:`\\star`.

    Parameters
    ----------
    omega : Cochain
        Primal :math:`k`-cochain.
    kind : {"diagonal", "galerkin"}, default "diagonal"
        Discretisation; see :cpp:enum:`declib::HodgeKind`.

    Returns
    -------
    Cochain
        Dual :math:`(n-k)`-cochain.
    """
    return _core.hodge_star(omega, kind)


def laplacian(omega: Cochain) -> Cochain:
    r"""Hodge Laplacian :math:`\Delta = d\delta + \delta d`.

    The codifferential is :math:`\delta = (-1)^{n(k+1)+1} \star d \star`.

    Parameters
    ----------
    omega : Cochain
        Primal :math:`k`-cochain.

    Returns
    -------
    Cochain
        Primal :math:`k`-cochain :math:`\Delta\omega`.
    """
    return _core.laplacian(omega)
