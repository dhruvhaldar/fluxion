import time
from fluxion.grid import StaggeredGrid
from fluxion.models import NavierStokes2D

grid = StaggeredGrid(nx=200, ny=200, lx=1.0, ly=1.0)
solver = NavierStokes2D(grid, re=100, dt=0.005)

# warmup
solver.step()

start = time.time()
for _ in range(50):
    solver.step()
end = time.time()
print(f"Original Time: {end - start:.4f} seconds")
