import numpy as np
import time
from fluxion.grid import StaggeredGrid
from fluxion.discretization import compute_laplacian

grid = StaggeredGrid(nx=200, ny=200, lx=1.0, ly=1.0)
phi = np.random.rand(grid.nx, grid.ny)

# warmup
compute_laplacian(phi, grid)

start = time.time()
for _ in range(5000):
    compute_laplacian(phi, grid)
end = time.time()
print(f"Time: {end - start:.4f} seconds")
