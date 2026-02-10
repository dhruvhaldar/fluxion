from fluxion.models import NavierStokes2D
from fluxion.grid import StaggeredGrid
import os

if __name__ == "__main__":
    print("Running Lid Driven Cavity (50x50)...")
    grid = StaggeredGrid(nx=50, ny=50, lx=1.0, ly=1.0)
    solver = NavierStokes2D(grid, re=100, dt=0.005)
    solver.set_boundary_condition('top', u=1.0)
    solver.solve(steps=2000)

    os.makedirs('assets', exist_ok=True)
    solver.plot_streamlines(save_path='assets/lid_driven_streamlines.png')
    print("Saved assets/lid_driven_streamlines.png")
