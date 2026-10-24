"""Quickstart: the Laplacian of a manufactured solution on the periodic unit square."""

import math

import declib as dl

topo = dl.StagTopology((64, 64))
geom = dl.UniformGeometry(topo, lengths=(1.0, 1.0))
M = dl.DiscreteManifold(topo, geom)

S, T = dl.Orientation.STRAIGHT, dl.Orientation.TWISTED
V0, V1, tV1 = M.space(0, S), M.space(1, S), M.space(1, T)

d0, d1 = dl.ExteriorDerivative(V0), dl.ExteriorDerivative(tV1)
s0, s1 = dl.HodgeStar(V0), dl.HodgeStar(V1)
lap = dl.inverse(s0) @ d1 @ s1 @ d0


def u_exact(x, y):
    return math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y)


u = V0.from_function(u_exact)
f = lap(u)  # approximately -8 pi^2 u
A = lap.assemble()  # petsc4py.PETSc.Mat

err = u.copy()
err *= -8 * math.pi**2
err -= f
print(f"max |lap u + 8 pi^2 u| / (8 pi^2) = {err.norm('max') / (8 * math.pi**2):.3e}")
