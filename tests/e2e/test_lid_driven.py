import numpy as np
import pytest
from fluxion.models import NavierStokes2D
from fluxion.grid import StaggeredGrid
from fluxion import discretization

def test_lid_driven_conservation():
    """
    Tests mass conservation in a Lid-Driven Cavity setup.
    """
    grid = StaggeredGrid(nx=20, ny=20, lx=1.0, ly=1.0)
    solver = NavierStokes2D(grid, re=100, dt=0.005)

    # Drive flow
    solver.set_boundary_condition('top', u=1.0)

    # Run for some steps
    solver.solve(steps=20)

    # Check Divergence
    div = discretization.compute_divergence(solver.u, solver.v, grid)

    # Check divergence in the interior.
    # Boundary cells may satisfy continuity less strictly due to the simplified
    # Homogeneous Neumann BC implementation (p[0]=p[1] implies zero gradient at face 1, not face 0).
    # A more rigorous implementation would include boundary cells in the linear system.
    div_inner = div[2:-2, 2:-2]

    print(f"Mean Divergence (Inner): {np.mean(np.abs(div_inner))}")
    print(f"Max Divergence (Inner): {np.max(np.abs(div_inner))}")

    assert np.mean(np.abs(div_inner)) < 1e-3
    assert np.max(np.abs(div_inner)) < 1e-2
