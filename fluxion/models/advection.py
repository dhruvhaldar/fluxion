import numpy as np
import matplotlib.pyplot as plt
from fluxion.grid import StaggeredGrid
from fluxion import discretization, time_stepping

class AdvectionDiffusion:
    """
    Solver for the scalar Advection-Diffusion equation:
    d(phi)/dt + div(u * phi) = nu * laplacian(phi)
    """
    def __init__(self, grid, u_field, v_field, nu=0.0):
        self.grid = grid
        self.u = u_field
        self.v = v_field
        self.nu = nu
        self.phi = np.zeros((grid.nx, grid.ny))
        self.scheme = 'central'

    def set_initial_condition(self, func):
        self.phi = self.grid.initialize_field(func, location='center')

    def compute_rhs(self, phi):
        # Convection
        conv = discretization.convection_term(phi, self.u, self.v, self.grid, scheme=self.scheme)

        # Diffusion
        if self.nu > 0:
            diff = discretization.compute_laplacian(phi, self.grid)
        else:
            diff = 0.0

        return (self.nu * diff) - conv

    def step(self, dt, scheme='central', method='euler'):
        """
        Advances the solution by one time step dt.
        """
        self.scheme = scheme

        if method == 'euler':
            rhs = self.compute_rhs(self.phi)
            self.phi = time_stepping.euler_step(self.phi, rhs, dt)
        elif method == 'rk4':
            self.phi = time_stepping.rk4_step(self.phi, self.compute_rhs, dt)
        else:
            raise ValueError(f"Unknown time stepping method: {method}")

    @classmethod
    def compare_schemes(cls, schemes=['upwind', 'central', 'quick'], save_path=None):
        """
        Runs a benchmark comparison of different convection schemes on a step profile.
        """
        # 1D-like problem setup
        nx = 100
        ny = 5 # Small y dimension
        grid = StaggeredGrid(nx, ny, lx=1.0, ly=0.05)

        # Constant velocity u=1, v=0
        u = np.ones((nx+1, ny))
        v = np.zeros((nx, ny+1))

        # Step Profile Initial Condition: phi=1 for x < 0.2, else 0
        def step_profile(x, y):
            val = np.zeros_like(x)
            val[x < 0.2] = 1.0
            return val

        dt = 0.001
        steps = 400 # t = 0.4. Center should move to 0.6

        results = {}

        for scheme in schemes:
            solver = cls(grid, u, v, nu=0.0) # Pure advection
            solver.set_initial_condition(step_profile)

            for _ in range(steps):
                # Use RK4 for time accuracy to isolate spatial errors
                solver.step(dt, scheme=scheme, method='rk4')

            # Extract 1D profile from center
            results[scheme] = solver.phi[:, ny//2].copy()

        # Plotting
        plt.figure(figsize=(10, 6))
        x = grid.x_c

        # Analytical Solution (Shifted step)
        # Original step at 0.2. Velocity 1.0. Time 0.4. New step at 0.6.
        y_exact = np.zeros_like(x)
        y_exact[x < 0.6] = 1.0
        plt.plot(x, y_exact, 'k--', label='Exact', linewidth=2)

        for scheme in schemes:
            plt.plot(x, results[scheme], label=scheme.upper())

        plt.title(f"Convection Scheme Comparison (t={steps*dt:.1f})")
        plt.xlabel("x")
        plt.ylabel("phi")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
