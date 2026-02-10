import numpy as np
import pytest
from fluxion.models import NavierStokes2D
from fluxion.grid import StaggeredGrid

def test_poiseuille_no_force():
    """
    Validates that without driving force or moving boundaries, the velocity remains zero.
    """
    grid = StaggeredGrid(nx=10, ny=10, lx=1.0, ly=1.0)
    solver = NavierStokes2D(grid, re=10, dt=0.01)

    # Run solver
    solver.solve(steps=10)

    # Velocity should be zero (within machine precision/numerical noise)
    assert np.allclose(solver.u, 0.0, atol=1e-10)
    assert np.allclose(solver.v, 0.0, atol=1e-10)

def test_poiseuille_moving_wall():
    """
    Validates that moving a wall induces flow (Couette-like check).
    """
    grid = StaggeredGrid(nx=10, ny=10, lx=1.0, ly=1.0)
    solver = NavierStokes2D(grid, re=10, dt=0.01)

    solver.set_boundary_condition('top', u=1.0)
    solver.solve(steps=10)

    # Top velocity should have propagated into the domain slightly
    assert np.max(np.abs(solver.u)) > 0.0
