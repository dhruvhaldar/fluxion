import time
from fluxion.grid import StaggeredGrid
from fluxion.models.navier_stokes import NavierStokes2D

grid = StaggeredGrid(nx=200, ny=200, lx=1.0, ly=1.0)
solver = NavierStokes2D(grid, re=100, dt=0.005)

start = time.time()
for _ in range(10):
    solver.step()
end = time.time()
print(f"Time: {end - start:.4f} seconds")
