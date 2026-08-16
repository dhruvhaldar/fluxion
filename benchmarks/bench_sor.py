import numpy as np
import time
from fluxion.grid import StaggeredGrid
from fluxion.solvers import LinearSolver

grid = StaggeredGrid(nx=200, ny=200, lx=1.0, ly=1.0)
p = np.zeros((grid.nx, grid.ny))
rhs = np.random.rand(grid.nx, grid.ny)

start = time.time()
LinearSolver.solve_sor(p, rhs, grid, max_iter=1000, tol=1e-10)
end = time.time()
print(f"Time: {end - start:.4f} seconds")
