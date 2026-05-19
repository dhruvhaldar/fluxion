import numpy as np
import time
from fluxion.grid import StaggeredGrid
from fluxion.discretization import convection_term

grid = StaggeredGrid(nx=200, ny=200, lx=1.0, ly=1.0)
phi = np.random.rand(grid.nx, grid.ny)
u = np.random.rand(grid.nx+1, grid.ny)
v = np.random.rand(grid.nx, grid.ny+1)

# warmup
convection_term(phi, u, v, grid)

start = time.time()
for _ in range(1000):
    convection_term(phi, u, v, grid)
end = time.time()
print(f"Time: {end - start:.4f} seconds")
