import numpy as np
import matplotlib.pyplot as plt
from fluxion.grid import StaggeredGrid
from fluxion import solvers
from fluxion import discretization

class NavierStokes2D:
    """
    2D Incompressible Navier-Stokes Solver using Chorin's Projection Method.
    Grid: Staggered.
    """
    def __init__(self, grid, re=100.0, dt=0.001, channel_flow=False):
        self.grid = grid
        self.re = re
        self.dt = dt
        self.nu = 1.0 / re

        # Fields
        self.u = np.zeros((grid.nx+1, grid.ny))
        self.v = np.zeros((grid.nx, grid.ny+1))
        self.p = np.zeros((grid.nx, grid.ny))

        self.channel_flow = channel_flow

        # Boundary conditions values
        self.bc_u = {'top': 0.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0}
        self.bc_v = {'top': 0.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0}

    def set_boundary_condition(self, side, u=None, v=None):
        if u is not None: self.bc_u[side] = u
        if v is not None: self.bc_v[side] = v

    def solve(self, steps):
        """
        Time stepping loop.
        """
        for step in range(steps):
            self.step()

    def solve_steady_state(self, tol=1e-4, max_steps=10000):
        """
        Run until convergence.
        """
        for step in range(max_steps):
            u_old = self.u.copy()
            self.step()
            diff = np.max(np.abs(self.u - u_old))
            if diff < tol:
                print(f"Converged in {step} steps.")
                break

    def step(self):
        """
        Single time step.
        """
        dt = self.dt
        grid = self.grid
        nx, ny = grid.nx, grid.ny
        dx, dy = grid.dx, grid.dy

        # 1. Prediction Step (Intermediate Velocity u*, v*)
        # RHS = - (u.grad)u + nu * lap(u)

        u = self.u
        v = self.v

        # --- U-Momentum ---
        # Compute derivatives for u (defined at i+1/2, j)
        # We loop over internal u-points: i=1..nx-1, j=0..ny-1

        u_star = u.copy()

        # Advection u * du/dx
        # Central difference
        du_dx = (u[1:-1, :] - u[:-2, :]) / (2*dx) # Wrong, u has nx+1 points.
        # indices: i-1, i, i+1.
        # Internal points i range from 1 to nx-1.
        # u[i+1] - u[i-1]

        # Correct Slicing for internal u (1 to nx-1)
        # i=1 corresponds to index 1. Left neighbor 0. Right neighbor 2.
        u_interior = u[1:-1, :]
        inv_2dx = 1.0 / (2*dx)
        inv_2dy = 1.0 / (2*dy)
        inv_dx2 = 1.0 / (dx**2)
        inv_dy2 = 1.0 / (dy**2)

        du_dx = (u[2:, :] - u[:-2, :]) * inv_2dx

        # Advection v * du/dy
        # Need v at u-locations.
        # v (nx, ny+1). u (nx+1, ny).
        # v at (i, j) corresponds to bottom face of cell (i, j).
        # u at (i, j) corresponds to left face of cell (i, j).
        # We need v at (i, j) of u-grid?
        # Interpolation: v_interp = average of 4 v neighbors.
        # v[i, j], v[i, j+1], v[i-1, j], v[i-1, j+1]

        # We need v at i=1..nx-1.
        # v array has x-dim nx. i index goes 0..nx-1.
        # For u[i, j], neighbors are v[i, j], v[i, j+1], v[i-1, j], v[i-1, j+1].
        # Indices in v array: i and i-1.

        # ⚡ Bolt: Use in-place operations to prevent implicit array creations for v_interp
        v_interp = np.empty_like(u_interior)
        np.add(v[:-1, 1:], v[1:, 1:], out=v_interp)
        np.add(v_interp, v[:-1, :-1], out=v_interp)
        np.add(v_interp, v[1:, :-1], out=v_interp)
        np.multiply(v_interp, 0.25, out=v_interp)

        # du/dy
        # u[i, j+1] - u[i, j-1]
        # We need u with ghost cells in y.
        # Since u has shape (nx+1, ny), we only have j=0..ny-1.
        # For j=0, need j=-1. For j=ny-1, need j=ny.
        # Handle boundaries later. For now assume internal j=1..ny-2.

        du_dy = np.empty_like(u_interior)
        # Interior Y (j=1..ny-2)
        du_dy[:, 1:-1] = (u_interior[:, 2:] - u_interior[:, :-2]) * inv_2dy

        # Diffusion d2u/dx2 + d2u/dy2
        d2u_dy2 = np.empty_like(u_interior)
        d2u_dy2[:, 1:-1] = (u_interior[:, 2:] - 2*u_interior[:, 1:-1] + u_interior[:, :-2]) * inv_dy2

        # Apply BCs for u derivatives near walls
        # Top Wall (Lid): u = u_lid.
        # d2u/dy2 at j=ny-1.
        # u_ghost = 2*u_top - u_last.
        # (u_ghost - 2*u_last + u_last_minus_1) / dy^2
        # = (2*u_top - u_last - 2*u_last + u_last_minus_1) / dy^2
        # = (2*u_top - 3*u_last + u_last_minus_1) / dy^2
        u_top = self.bc_u['top']
        d2u_dy2[:, -1] = (2*u_top - 3*u_interior[:, -1] + u_interior[:, -2]) * inv_dy2

        u_bot = self.bc_u['bottom']
        d2u_dy2[:, 0] = (2*u_bot - 3*u_interior[:, 0] + u_interior[:, 1]) * inv_dy2

        # Advection needs full du_dy?
        # du/dy at j=ny-1: (u_ghost - u_last_minus_1) / 2dy
        # = (2*u_top - u_last - u_last_minus_1) / 2dy
        du_dy[:, -1] = (2*u_top - u_interior[:, -1] - u_interior[:, -2]) * inv_2dy
        du_dy[:, 0] = (u_interior[:, 1] - (2*u_bot - u_interior[:, 0])) * inv_2dy

        # ⚡ Bolt: Chain inplace operations to avoid implicit array allocations
        rhs_u = np.empty_like(u_interior)

        # ⚡ Bolt: Prevent implicit allocations in advection term multiplications
        tmp_u = np.empty_like(u_interior)

        np.multiply(u[1:-1, :], 2.0, out=tmp_u)
        np.subtract(u[2:, :], tmp_u, out=rhs_u)
        np.add(rhs_u, u[:-2, :], out=rhs_u)
        np.multiply(rhs_u, inv_dx2, out=rhs_u)

        rhs_u += d2u_dy2
        rhs_u *= self.nu

        np.multiply(u_interior, du_dx, out=tmp_u)
        rhs_u -= tmp_u
        np.multiply(v_interp, du_dy, out=tmp_u)
        rhs_u -= tmp_u

        u_star[1:-1, :] = u_interior + dt * rhs_u

        # --- V-Momentum ---
        v_star = v.copy()
        v_interior = v[:, 1:-1] # j=1..ny-1

        # dv/dy
        dv_dx = np.empty_like(v_interior)

        # d2v/dy2
        d2v_dx2 = np.empty_like(v_interior)

        # Boundaries for V (Left/Right)
        v_left = self.bc_v['left']
        v_right = self.bc_v['right']

        # x=0 (i=0):
        # d2v/dx2 = (v[1] - 2v[0] + v_ghost) / dx^2
        # v_ghost = 2*v_left - v[0]
        # => (v[1] - 3v[0] + 2*v_left) / dx^2
        d2v_dx2[0, :] = (v_interior[1, :] - 3*v_interior[0, :] + 2*v_left) * inv_dx2
        d2v_dx2[-1, :] = (2*v_right - 3*v_interior[-1, :] + v_interior[-2, :]) * inv_dx2
        d2v_dx2[1:-1, :] = (v_interior[2:, :] - 2*v_interior[1:-1, :] + v_interior[:-2, :]) * inv_dx2

        dv_dx[0, :] = (v_interior[1, :] - (2*v_left - v_interior[0, :])) * inv_2dx
        dv_dx[-1, :] = ((2*v_right - v_interior[-1, :]) - v_interior[-2, :]) * inv_2dx
        dv_dx[1:-1, :] = (v_interior[2:, :] - v_interior[:-2, :]) * inv_2dx

        # u interpolation for v eq
        # u at (i, j), (i+1, j), (i, j-1), (i+1, j-1)
        # v interior j ranges 1..ny-1.
        # We need u around v[i, j]

        # ⚡ Bolt: Use in-place operations to prevent implicit array creations for u_interp
        u_interp = np.empty_like(v_interior)
        np.add(u[:-1, :-1], u[1:, :-1], out=u_interp)
        np.add(u_interp, u[:-1, 1:], out=u_interp)
        np.add(u_interp, u[1:, 1:], out=u_interp)
        np.multiply(u_interp, 0.25, out=u_interp)

        # ⚡ Bolt: Chain inplace operations to avoid implicit array allocations
        rhs_v = d2v_dx2

        # ⚡ Bolt: Prevent implicit allocations in advection term multiplications
        tmp_v = np.empty_like(v_interior)

        np.multiply(v[:, 1:-1], 2.0, out=tmp_v)
        np.subtract(v[:, 2:], tmp_v, out=tmp_v)
        np.add(tmp_v, v[:, :-2], out=tmp_v)
        np.multiply(tmp_v, inv_dy2, out=tmp_v)

        rhs_v += tmp_v
        rhs_v *= self.nu

        np.multiply(u_interp, dv_dx, out=tmp_v)
        rhs_v -= tmp_v

        # v_interior * ((v[:, 2:] - v[:, :-2]) * inv_2dy)
        np.subtract(v[:, 2:], v[:, :-2], out=tmp_v)
        np.multiply(tmp_v, inv_2dy, out=tmp_v)
        np.multiply(v_interior, tmp_v, out=tmp_v)
        rhs_v -= tmp_v

        v_star[:, 1:-1] = v_interior + dt * rhs_v

        # Enforce BCs on Intermediate Velocity (Walls)
        # Tangential velocities
        u_star[0, :] = self.bc_u['left']
        u_star[-1, :] = self.bc_u['right']
        v_star[:, 0] = self.bc_v['bottom']
        v_star[:, -1] = self.bc_v['top']

        # 2. Pressure Solve
        # div(u*) / dt = Lap(p)
        div_u_star = discretization.compute_divergence(u_star, v_star, grid)

        # ⚡ Bolt: Perform in-place division to avoid allocating a new array for rhs_p
        div_u_star /= dt

        # Solve PPE
        self.p, _ = solvers.LinearSolver.solve_jacobi(self.p, div_u_star, grid, max_iter=2000, tol=1e-5)

        # 3. Correction
        # u = u* - dt * grad(p)
        grad_p_x, grad_p_y = discretization.compute_gradient(self.p, grid)

        self.u = u_star - dt * grad_p_x
        self.v = v_star - dt * grad_p_y

        # Enforce BCs again?
        # Projection method naturally enforces div(u)=0.
        # But tangential BCs might drift if grad(p) is not perfect at boundaries.
        # Usually we re-enforce Dirichlet BCs.
        self.u[0, :] = self.bc_u['left']
        self.u[-1, :] = self.bc_u['right']
        self.v[:, 0] = self.bc_v['bottom']
        self.v[:, -1] = self.bc_v['top']

    def plot_streamlines(self, save_path=None):
        """
        Plots streamlines of the velocity field.
        """
        # Interpolate u and v to cell centers for plotting
        u_c = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        v_c = 0.5 * (self.v[:, :-1] + self.v[:, 1:])

        speed = np.sqrt(u_c**2 + v_c**2)

        plt.figure(figsize=(8, 6))

        X, Y = self.grid.X_c, self.grid.Y_c

        # Streamplot requires meshgrid
        plt.streamplot(X.T, Y.T, u_c.T, v_c.T, color='k', density=1.5, linewidth=0.5)
        plt.contourf(X.T, Y.T, speed.T, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Velocity Magnitude')
        plt.title(f"Streamlines (Re={self.re})")
        plt.xlabel("x")
        plt.ylabel("y")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
