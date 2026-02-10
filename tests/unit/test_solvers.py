import numpy as np
import pytest
from fluxion.grid import StaggeredGrid
from fluxion import solvers, discretization

def test_jacobi_solver_consistency():
    """
    Checks if the Jacobi solver returns a field whose Laplacian matches the RHS.
    Note: For Neumann problems, p is unique up to a constant, but Lap(p) should match RHS.
    """
    grid = StaggeredGrid(nx=30, ny=30)
    # RHS field: sin(2*pi*x)*sin(2*pi*y)
    # This function has integral roughly 0, which is good for Neumann solvability condition.
    rhs = np.sin(2*np.pi * grid.X_c) * np.sin(2*np.pi * grid.Y_c)

    # Solve Lap(p) = rhs
    p_initial = np.zeros((grid.nx, grid.ny))
    p_sol, it = solvers.LinearSolver.solve_jacobi(p_initial, rhs, grid, max_iter=5000, tol=1e-6)

    # Compute Laplacian of result
    lap_p = discretization.compute_laplacian(p_sol, grid)

    # Error should be small in the interior
    # Excluding boundaries where Laplacian stencil is not valid in our implementation
    error = np.abs(lap_p[2:-2, 2:-2] - rhs[2:-2, 2:-2])

    assert np.mean(error) < 0.1

def test_sor_solver_consistency():
    """
    Checks if the SOR solver returns a field whose Laplacian matches the RHS.
    """
    grid = StaggeredGrid(nx=30, ny=30)
    rhs = np.cos(2*np.pi * grid.X_c) * np.cos(2*np.pi * grid.Y_c)

    p_initial = np.zeros((grid.nx, grid.ny))
    p_sol, it = solvers.LinearSolver.solve_sor(p_initial, rhs, grid, omega=1.8, max_iter=5000, tol=1e-6)

    lap_p = discretization.compute_laplacian(p_sol, grid)

    # Interior check
    error = np.abs(lap_p[2:-2, 2:-2] - rhs[2:-2, 2:-2])
    assert np.mean(error) < 0.1
