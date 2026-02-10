import numpy as np
import matplotlib.pyplot as plt
from fluxion.models import NavierStokes2D
from fluxion.grid import StaggeredGrid
import os

def run_simulation(nx):
    """
    Runs lid driven cavity for nx x nx grid.
    Returns u-velocity along vertical centerline.
    """
    grid = StaggeredGrid(nx=nx, ny=nx, lx=1.0, ly=1.0)
    solver = NavierStokes2D(grid, re=100, dt=0.005)
    solver.set_boundary_condition('top', u=1.0)

    # Run to approximate steady state or fixed time
    # 1000 steps at dt=0.005 is T=5.0
    solver.solve(steps=1000)

    # Extract u at x=0.5 (index nx//2 for u-faces)
    # u is (nx+1, ny). x-coordinates are 0, dx, 2dx...
    # index nx//2 corresponds to x=0.5.
    u_centerline = solver.u[nx//2, :]
    return u_centerline, grid.y_c

def main():
    resolutions = [16, 32, 64]
    dxs = [1.0/n for n in resolutions]

    u_profiles = []
    y_coords = []

    print("Running Grid Convergence Study...")

    for nx in resolutions:
        print(f"  Simulating Grid: {nx}x{nx}...")
        u, y = run_simulation(nx)
        u_profiles.append(u)
        y_coords.append(y)

    # Compute error relative to finest grid (64)
    # Interpolate coarse results onto fine grid y-coordinates
    fine_u = u_profiles[-1]
    fine_y = y_coords[-1]

    errors = []

    # Compare 16 and 32 against 64
    for i in range(len(resolutions)-1):
        # Interpolate coarse u onto fine y
        u_interp = np.interp(fine_y, y_coords[i], u_profiles[i])

        # L2 Error Norm
        l2_err = np.sqrt(np.mean((u_interp - fine_u)**2))
        errors.append(l2_err)

    # Plot Convergence
    plt.figure(figsize=(8, 6))
    plt.loglog(dxs[:-1], errors, 'o-', label='Simulation Error (vs Fine Grid)')

    # Reference Slope 2
    # Fit line through last point (smallest error)
    ref_x = np.array(dxs[:-1])
    # Assume error ~ C * dx^2
    # errors[-1] = C * dxs[-2]^2 -> C = errors[-1] / dxs[-2]^2
    # y = C * x^2
    C = errors[-1] / (dxs[-2]**2)
    ref_y = C * (ref_x**2)

    plt.loglog(ref_x, ref_y, 'k--', label='2nd Order Slope')

    plt.xlabel('Grid Spacing (dx)')
    plt.ylabel('L2 Error Norm')
    plt.title('Grid Convergence Study (Lid Driven Cavity Re=100)')
    plt.legend()
    plt.grid(True, which="both", ls="-")

    os.makedirs('assets', exist_ok=True)
    save_path = 'assets/grid_convergence.png'
    plt.savefig(save_path)
    print(f"Convergence plot saved to {save_path}")

if __name__ == "__main__":
    main()
