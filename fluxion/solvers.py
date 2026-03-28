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
        p_old = p.copy()

        # Pre-calculate factors to avoid repeated division in the loop
        mult_x = 1.0 / (dx2 * denom)
        mult_y = 1.0 / (dy2 * denom)
        rhs_scaled = rhs[1:-1, 1:-1] / denom

        # ⚡ Bolt: Pre-allocate a temporary array to avoid implicit array creations in the loop
        tmp_y = np.zeros_like(rhs_scaled)

        check_interval = 50

        for it in range(max_iter):
            # Swap references instead of allocating a new array
            p_old, p_new = p_new, p_old

            # ⚡ Bolt: Fully in-place interior update to eliminate implicit temporary arrays
            # X-direction directly into p_new
            np.add(p_old[2:, 1:-1], p_old[:-2, 1:-1], out=p_new[1:-1, 1:-1])
            np.multiply(p_new[1:-1, 1:-1], mult_x, out=p_new[1:-1, 1:-1])

            # Y-direction into tmp_y
            np.add(p_old[1:-1, 2:], p_old[1:-1, :-2], out=tmp_y)
            np.multiply(tmp_y, mult_y, out=tmp_y)

            # Combine
            np.add(p_new[1:-1, 1:-1], tmp_y, out=p_new[1:-1, 1:-1])
            np.subtract(p_new[1:-1, 1:-1], rhs_scaled, out=p_new[1:-1, 1:-1])

            # Boundary Conditions (Homogeneous Neumann)
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]

            # Only calculate max diff every check_interval to avoid expensive array operations
            if it % check_interval == 0 and it > 0:
                if np.max(np.abs(p_new - p_old)) < tol:
                    return p_new, it

        # Final check if loop finishes
        if np.max(np.abs(p_new - p_old)) < tol:
            return p_new, max_iter - 1

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
        i_idx, j_idx = np.meshgrid(np.arange(1, nx-1), np.arange(1, ny-1), indexing='ij')
        mask_red = (i_idx + j_idx) % 2 == 0
        mask_black = (i_idx + j_idx) % 2 == 1

        # Pre-calculate factors for inside loop
        mult_x = omega / (dx2 * denom)
        mult_y = omega / (dy2 * denom)
        rhs_scaled = omega * rhs[1:-1, 1:-1] / denom

        p_slice = p_new[1:-1, 1:-1]

        check_interval = 50
        p_old = p_new.copy()

        for it in range(max_iter):
            # Capture the previous iteration's state right before the check iteration
            if it > 0 and it % check_interval == 0:
                np.copyto(p_old, p_new)

            # 1. Update Red Points
            # ⚡ Bolt: Use in-place operators to avoid implicit temporary whole-array creation
            p_gs_red = p_new[2:, 1:-1] + p_new[:-2, 1:-1]
            p_gs_red *= mult_x

            tmp_y = p_new[1:-1, 2:] + p_new[1:-1, :-2]
            tmp_y *= mult_y

            p_gs_red += tmp_y
            p_gs_red -= rhs_scaled

            # Update only Red points in-place using np.putmask for performance
            # Add (1 - omega) * p_slice in-place to avoid implicit temporary whole-array creation
            if omega != 1.0:
                p_gs_red += (1 - omega) * p_slice
            np.putmask(p_slice, mask_red, p_gs_red)

            # 2. Update Black Points
            # Recompute neighbors (Red points have changed)
            p_gs_black = p_new[2:, 1:-1] + p_new[:-2, 1:-1]
            p_gs_black *= mult_x

            tmp_y = p_new[1:-1, 2:] + p_new[1:-1, :-2]
            tmp_y *= mult_y

            p_gs_black += tmp_y
            p_gs_black -= rhs_scaled

            # Update only Black points in-place
            if omega != 1.0:
                p_gs_black += (1 - omega) * p_slice
            np.putmask(p_slice, mask_black, p_gs_black)

            # Boundary Conditions
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]

            # Only calculate max diff every check_interval to avoid expensive array operations
            # We captured p_old at the START of this iteration, so p_new - p_old is exactly the diff
            # of this one iteration.
            if it > 0 and it % check_interval == 0:
                if np.max(np.abs(p_new - p_old)) < tol:
                    return p_new, it

        # Final check if loop finishes
        # Check against previous iteration which wasn't saved, so we just return.
        # It's an edge case, we can assume it hit max_iter without converging.
        return p_new, max_iter
