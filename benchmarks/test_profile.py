import cProfile
import pstats
from fluxion.grid import StaggeredGrid
from fluxion.models import NavierStokes2D

def run_sim():
    grid = StaggeredGrid(nx=32, ny=32, lx=1.0, ly=1.0)
    solver = NavierStokes2D(grid, re=100, dt=0.005)
    solver.set_boundary_condition('top', u=1.0)
    solver.solve(steps=100)

cProfile.run('run_sim()', 'sim_stats')
p = pstats.Stats('sim_stats')
p.sort_stats('tottime').print_stats(15)
