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

        p1 = p.copy()
        p2 = p.copy()

        # Pre-calculate factors to avoid repeated division in the loop
        mult_x = 1.0 / (dx2 * denom)
        mult_y = 1.0 / (dy2 * denom)
        rhs_scaled = rhs[1:-1, 1:-1] / denom

        # ⚡ Bolt: Factor out mult_x to reduce total array operations in the loop.
        # p_new = mult_x * (p_up + p_down) + mult_y * (p_right + p_left) - rhs_scaled
        # becomes: p_new = mult_x * [ (p_up + p_down) + (mult_y/mult_x) * (p_right + p_left) - (rhs_scaled/mult_x) ]
        mult_y_over_x = mult_y / mult_x
        rhs_eff = rhs_scaled / mult_x

        # ⚡ Bolt: Pre-allocate a temporary array to avoid implicit array creations in the loop
        tmp_y = np.zeros_like(rhs_scaled)

        # ⚡ Bolt: Pre-compute slice views outside loops to eliminate slicing overhead.
        # These views remain valid as the arrays are updated in place.
        p1_up = p1[2:, 1:-1]
        p1_down = p1[:-2, 1:-1]
        p1_right = p1[1:-1, 2:]
        p1_left = p1[1:-1, :-2]
        p1_center = p1[1:-1, 1:-1]

        p2_up = p2[2:, 1:-1]
        p2_down = p2[:-2, 1:-1]
        p2_right = p2[1:-1, 2:]
        p2_left = p2[1:-1, :-2]
        p2_center = p2[1:-1, 1:-1]

        check_interval = 50

        p_old, p_new = p1, p2
        p_old_up, p_old_down, p_old_right, p_old_left = p1_up, p1_down, p1_right, p1_left
        p_new_center = p2_center

        for it in range(max_iter):
            # ⚡ Bolt: Swap references of pre-computed slice views
            if it % 2 == 0:
                p_old, p_new = p1, p2
                p_old_up, p_old_down, p_old_right, p_old_left = p1_up, p1_down, p1_right, p1_left
                p_new_center = p2_center
            else:
                p_old, p_new = p2, p1
                p_old_up, p_old_down, p_old_right, p_old_left = p2_up, p2_down, p2_right, p2_left
                p_new_center = p1_center

            # ⚡ Bolt: Fully in-place interior update to eliminate implicit temporary arrays
            # Factoring out mult_x reduces the number of operations per iteration by combining terms.
            np.add(p_old_right, p_old_left, out=tmp_y)
            if mult_y_over_x != 1.0:
                np.multiply(tmp_y, mult_y_over_x, out=tmp_y)
            np.add(tmp_y, p_old_up, out=tmp_y)
            np.add(tmp_y, p_old_down, out=tmp_y)
            np.subtract(tmp_y, rhs_eff, out=tmp_y)
            np.multiply(tmp_y, mult_x, out=p_new_center)

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

        # ⚡ Bolt: Factor out mult_x to reduce total array operations in the SOR loop.
        mult_y_over_x = mult_y / mult_x
        rhs_eff = rhs_scaled / mult_x

        p_slice = p_new[1:-1, 1:-1]

        check_interval = 50
        p_old = p_new.copy()

        # Pre-allocate temporary arrays to avoid implicit array creations in the loop
        # We need an array for the entire right-hand-side expression of the SOR update
        # p_gs_red and p_gs_black can share the same buffer since they are updated sequentially
        p_gs = np.zeros_like(p_slice)
        tmp_y = np.zeros_like(p_slice)

        # Pre-compute slice views to avoid overhead inside the loop.
        # These are views into p_new, which is updated in place, so they
        # stay valid as long as p_new's reference remains unchanged.
        p_up = p_new[2:, 1:-1]
        p_down = p_new[:-2, 1:-1]
        p_right = p_new[1:-1, 2:]
        p_left = p_new[1:-1, :-2]

        for it in range(max_iter):
            # Capture the previous iteration's state right before the check iteration
            if it > 0 and it % check_interval == 0:
                np.copyto(p_old, p_new)

            # 1. Update Red Points
            # ⚡ Bolt: Use factored in-place operators to avoid implicit temporary whole-array creation
            # and minimize numpy ufunc overhead.
            np.add(p_right, p_left, out=p_gs)
            if mult_y_over_x != 1.0:
                np.multiply(p_gs, mult_y_over_x, out=p_gs)
            np.add(p_gs, p_up, out=p_gs)
            np.add(p_gs, p_down, out=p_gs)
            np.subtract(p_gs, rhs_eff, out=p_gs)
            np.multiply(p_gs, mult_x, out=p_gs)

            # Update only Red points in-place using np.putmask for performance
            # Add (1 - omega) * p_slice in-place to avoid implicit temporary whole-array creation
            if omega != 1.0:
                # Add (1 - omega) * p_slice directly into p_gs
                # using a temporary buffer if necessary, or just one extra op
                np.add(p_gs, (1 - omega) * p_slice, out=p_gs)
            np.putmask(p_slice, mask_red, p_gs)

            # 2. Update Black Points
            # Recompute neighbors (Red points have changed)
            np.add(p_right, p_left, out=p_gs)
            if mult_y_over_x != 1.0:
                np.multiply(p_gs, mult_y_over_x, out=p_gs)
            np.add(p_gs, p_up, out=p_gs)
            np.add(p_gs, p_down, out=p_gs)
            np.subtract(p_gs, rhs_eff, out=p_gs)
            np.multiply(p_gs, mult_x, out=p_gs)

            # Update only Black points in-place
            if omega != 1.0:
                np.add(p_gs, (1 - omega) * p_slice, out=p_gs)
            np.putmask(p_slice, mask_black, p_gs)

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
