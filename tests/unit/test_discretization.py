import numpy as np
import pytest
from fluxion.grid import StaggeredGrid
from fluxion import discretization

def test_divergence_linear():
    """
    Checks if divergence of linear velocity field u=x, v=y is correctly computed as 2.
    """
    grid = StaggeredGrid(nx=10, ny=10)
    # u = x (at u-faces)
    u = grid.X_u
    # v = y (at v-faces)
    v = grid.Y_v

    div = discretization.compute_divergence(u, v, grid)
    assert np.allclose(div, 2.0)

def test_laplacian_quadratic():
    """
    Checks if Laplacian of phi = x^2 + y^2 is 4.
    """
    grid = StaggeredGrid(nx=20, ny=20)
    phi = grid.X_c**2 + grid.Y_c**2
    lap = discretization.compute_laplacian(phi, grid)

    # Check only interior points where stencil is valid
    assert np.allclose(lap[1:-1, 1:-1], 4.0)

def test_gradient_linear():
    """
    Checks if gradient of p = 2x + 3y is [2, 3].
    """
    grid = StaggeredGrid(nx=10, ny=10)
    p = 2*grid.X_c + 3*grid.Y_c

    grad_x, grad_y = discretization.compute_gradient(p, grid)

    # Check interior faces
    assert np.allclose(grad_x[1:-1, :], 2.0)
    assert np.allclose(grad_y[:, 1:-1], 3.0)
