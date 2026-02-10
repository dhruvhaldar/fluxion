import numpy as np

class LinearSolver:
    @staticmethod
    def solve_jacobi(p, rhs, grid, max_iter=5000, tol=1e-5):
        """
        Solves Laplacian(p) = rhs using Jacobi Iteration.
        """
        dx, dy = grid.dx, grid.dy
        dx2, dy2 = dx**2, dy**2
        denom = 2 * (1/dx2 + 1/dy2)

        p_new = p.copy()

        for it in range(max_iter):
            p_old = p_new.copy()

            # Interior Update
            p_new[1:-1, 1:-1] = (
                (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) / dx2 +
                (p_old[1:-1, 2:] + p_old[1:-1, :-2]) / dy2 -
                rhs[1:-1, 1:-1]
            ) / denom

            # Boundary Conditions (Homogeneous Neumann)
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]

            if np.max(np.abs(p_new - p_old)) < tol:
                return p_new, it

        return p_new, max_iter

    @staticmethod
    def solve_sor(p, rhs, grid, omega=1.7, max_iter=5000, tol=1e-5):
        """
        Solves Laplacian(p) = rhs using Red-Black SOR.
        """
        dx, dy = grid.dx, grid.dy
        dx2, dy2 = dx**2, dy**2
        denom = 2 * (1/dx2 + 1/dy2)

        p_new = p.copy()
        nx, ny = grid.nx, grid.ny

        # Checkerboard masks for interior (1:-1, 1:-1)
        # Note: We need masks relative to the full array to map back correctly,
        # or relative to the slice.
        # Let's use masks for the slice to avoid index confusion.
        # Slice shape: (nx-2, ny-2)
        # Global indices: i from 1 to nx-2, j from 1 to ny-2.
        # (i+j) parity determines color.

        i_idx, j_idx = np.meshgrid(np.arange(1, nx-1), np.arange(1, ny-1), indexing='ij')
        mask_red = (i_idx + j_idx) % 2 == 0
        mask_black = (i_idx + j_idx) % 2 == 1

        for it in range(max_iter):
            p_old = p_new.copy()

            # 1. Update Red Points
            # Compute neighbors using current state
            neighbors = (
                (p_new[2:, 1:-1] + p_new[:-2, 1:-1]) / dx2 +
                (p_new[1:-1, 2:] + p_new[1:-1, :-2]) / dy2 -
                rhs[1:-1, 1:-1]
            )
            p_gs = neighbors / denom

            # Update only Red points
            p_new[1:-1, 1:-1][mask_red] = (1 - omega) * p_new[1:-1, 1:-1][mask_red] + omega * p_gs[mask_red]

            # 2. Update Black Points
            # Recompute neighbors (Red points have changed)
            neighbors = (
                (p_new[2:, 1:-1] + p_new[:-2, 1:-1]) / dx2 +
                (p_new[1:-1, 2:] + p_new[1:-1, :-2]) / dy2 -
                rhs[1:-1, 1:-1]
            )
            p_gs = neighbors / denom

            # Update only Black points
            p_new[1:-1, 1:-1][mask_black] = (1 - omega) * p_new[1:-1, 1:-1][mask_black] + omega * p_gs[mask_black]

            # Boundary Conditions
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]

            if np.max(np.abs(p_new - p_old)) < tol:
                return p_new, it

        return p_new, max_iter
