import numpy as np
import time
from fluxion.grid import StaggeredGrid
from fluxion.discretization import compute_gradient

grid = StaggeredGrid(nx=200, ny=200, lx=1.0, ly=1.0)
phi = np.random.rand(grid.nx, grid.ny)

# warmup
compute_gradient(phi, grid)

start = time.time()
for _ in range(1000):
    compute_gradient(phi, grid)
end = time.time()
print(f"Time: {end - start:.4f} seconds")
