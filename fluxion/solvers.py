import numpy as np

class LinearSolver:
    @staticmethod
    def solve_jacobi(p, rhs, grid, max_iter=5000, tol=1e-5):
        """
        Solves Laplacian(p) = rhs using Jacobi Iteration.
        """
        nx, ny = grid.nx, grid.ny
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

        check_interval = 200

        tmp_full = np.empty(p1.shape)

        # ⚡ Bolt: Unroll by 2 to avoid python variable assignment/swapping loop overhead.
        # Also hoist mult_y_over_x check out of the loop since it is invariant.
        if mult_y_over_x == 1.0:
            for it in range(0, max_iter, 2):
                # ⚡ Bolt: Replace chained in-place operations with inline math evaluations.
                p2_center[:] = (p1_right + p1_left + p1_up + p1_down - rhs_eff) * mult_x

                # ⚡ Bolt: Direct row assignments are slightly faster than slice assignments
                p2[0] = p2[1]
                p2[-1] = p2[-2]
                p2[:, 0] = p2[:, 1]
                p2[:, -1] = p2[:, -2]

                p1_center[:] = (p2_right + p2_left + p2_up + p2_down - rhs_eff) * mult_x

                p1[0] = p1[1]
                p1[-1] = p1[-2]
                p1[:, 0] = p1[:, 1]
                p1[:, -1] = p1[:, -2]

                if it > 0 and it % check_interval == 0:
                    np.subtract(p1, p2, out=tmp_full)
                    np.abs(tmp_full, out=tmp_full)
                    if np.max(tmp_full) < tol:
                        return p1, it + 1
        else:
            for it in range(0, max_iter, 2):
                # ⚡ Bolt: Replace chained in-place operations with inline math evaluations.
                p2_center[:] = ((p1_right + p1_left) * mult_y_over_x + p1_up + p1_down - rhs_eff) * mult_x

                p2[0] = p2[1]
                p2[-1] = p2[-2]
                p2[:, 0] = p2[:, 1]
                p2[:, -1] = p2[:, -2]

                p1_center[:] = ((p2_right + p2_left) * mult_y_over_x + p2_up + p2_down - rhs_eff) * mult_x

                p1[0] = p1[1]
                p1[-1] = p1[-2]
                p1[:, 0] = p1[:, 1]
                p1[:, -1] = p1[:, -2]

                if it > 0 and it % check_interval == 0:
                    np.subtract(p1, p2, out=tmp_full)
                    np.abs(tmp_full, out=tmp_full)
                    if np.max(tmp_full) < tol:
                        return p1, it + 1

        # Final check if loop finishes
        if np.max(np.abs(p1 - p2)) < tol:
            return p1, max_iter - 1

        return p1, max_iter

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

        # Pre-calculate factors for inside loop
        mult_x = omega / (dx2 * denom)
        mult_y = omega / (dy2 * denom)
        rhs_scaled = omega * rhs[1:-1, 1:-1] / denom

        # ⚡ Bolt: Factor out mult_x to reduce total array operations in the SOR loop.
        mult_y_over_x = mult_y / mult_x
        rhs_eff = rhs_scaled / mult_x

        p_slice = p_new[1:-1, 1:-1]

        check_interval = 200
        p_old = p_new.copy()

        tmp_full = np.empty(p.shape)

        # Pre-compute slice views to avoid overhead inside the loop.
        # These are views into p_new, which is updated in place, so they
        # stay valid as long as p_new's reference remains unchanged.
        p_up = p_new[2:, 1:-1]
        p_down = p_new[:-2, 1:-1]
        p_right = p_new[1:-1, 2:]
        p_left = p_new[1:-1, :-2]

        # ⚡ Bolt: Pre-calculate 1 - omega outside the loop
        one_minus_omega = 1.0 - omega

        # ⚡ Bolt: Hoist invariant branch conditions outside the loop to avoid Python overhead
        if mult_y_over_x == 1.0 and omega == 1.0:
            for it in range(max_iter):
                if it > 0 and it % check_interval == 0: np.copyto(p_old, p_new)

                # ⚡ Bolt: Replace np.putmask and boolean mask arrays with standard strided slicing
                # (e.g., [0::2, 0::2]) and inline vector math to eliminate implicit memory allocations and full-grid evaluations.
                p_slice[0::2, 0::2] = (p_right[0::2, 0::2] + p_left[0::2, 0::2] + p_up[0::2, 0::2] + p_down[0::2, 0::2] - rhs_eff[0::2, 0::2]) * mult_x
                p_slice[1::2, 1::2] = (p_right[1::2, 1::2] + p_left[1::2, 1::2] + p_up[1::2, 1::2] + p_down[1::2, 1::2] - rhs_eff[1::2, 1::2]) * mult_x

                p_slice[0::2, 1::2] = (p_right[0::2, 1::2] + p_left[0::2, 1::2] + p_up[0::2, 1::2] + p_down[0::2, 1::2] - rhs_eff[0::2, 1::2]) * mult_x
                p_slice[1::2, 0::2] = (p_right[1::2, 0::2] + p_left[1::2, 0::2] + p_up[1::2, 0::2] + p_down[1::2, 0::2] - rhs_eff[1::2, 0::2]) * mult_x

                p_new[0] = p_new[1]
                p_new[-1] = p_new[-2]
                p_new[:, 0] = p_new[:, 1]
                p_new[:, -1] = p_new[:, -2]

                if it > 0 and it % check_interval == 0:
                    np.subtract(p_new, p_old, out=tmp_full)
                    np.abs(tmp_full, out=tmp_full)
                    if np.max(tmp_full) < tol: return p_new, it

        elif mult_y_over_x == 1.0 and omega != 1.0:
            for it in range(max_iter):
                if it > 0 and it % check_interval == 0: np.copyto(p_old, p_new)

                # ⚡ Bolt: Replace np.putmask and boolean mask arrays with standard strided slicing
                p_slice[0::2, 0::2] = (p_right[0::2, 0::2] + p_left[0::2, 0::2] + p_up[0::2, 0::2] + p_down[0::2, 0::2] - rhs_eff[0::2, 0::2]) * mult_x + p_slice[0::2, 0::2] * one_minus_omega
                p_slice[1::2, 1::2] = (p_right[1::2, 1::2] + p_left[1::2, 1::2] + p_up[1::2, 1::2] + p_down[1::2, 1::2] - rhs_eff[1::2, 1::2]) * mult_x + p_slice[1::2, 1::2] * one_minus_omega

                p_slice[0::2, 1::2] = (p_right[0::2, 1::2] + p_left[0::2, 1::2] + p_up[0::2, 1::2] + p_down[0::2, 1::2] - rhs_eff[0::2, 1::2]) * mult_x + p_slice[0::2, 1::2] * one_minus_omega
                p_slice[1::2, 0::2] = (p_right[1::2, 0::2] + p_left[1::2, 0::2] + p_up[1::2, 0::2] + p_down[1::2, 0::2] - rhs_eff[1::2, 0::2]) * mult_x + p_slice[1::2, 0::2] * one_minus_omega

                p_new[0] = p_new[1]
                p_new[-1] = p_new[-2]
                p_new[:, 0] = p_new[:, 1]
                p_new[:, -1] = p_new[:, -2]

                if it > 0 and it % check_interval == 0:
                    np.subtract(p_new, p_old, out=tmp_full)
                    np.abs(tmp_full, out=tmp_full)
                    if np.max(tmp_full) < tol: return p_new, it

        elif mult_y_over_x != 1.0 and omega == 1.0:
            for it in range(max_iter):
                if it > 0 and it % check_interval == 0: np.copyto(p_old, p_new)

                # ⚡ Bolt: Replace np.putmask and boolean mask arrays with standard strided slicing
                p_slice[0::2, 0::2] = ( (p_right[0::2, 0::2] + p_left[0::2, 0::2]) * mult_y_over_x + p_up[0::2, 0::2] + p_down[0::2, 0::2] - rhs_eff[0::2, 0::2] ) * mult_x
                p_slice[1::2, 1::2] = ( (p_right[1::2, 1::2] + p_left[1::2, 1::2]) * mult_y_over_x + p_up[1::2, 1::2] + p_down[1::2, 1::2] - rhs_eff[1::2, 1::2] ) * mult_x

                p_slice[0::2, 1::2] = ( (p_right[0::2, 1::2] + p_left[0::2, 1::2]) * mult_y_over_x + p_up[0::2, 1::2] + p_down[0::2, 1::2] - rhs_eff[0::2, 1::2] ) * mult_x
                p_slice[1::2, 0::2] = ( (p_right[1::2, 0::2] + p_left[1::2, 0::2]) * mult_y_over_x + p_up[1::2, 0::2] + p_down[1::2, 0::2] - rhs_eff[1::2, 0::2] ) * mult_x

                p_new[0] = p_new[1]
                p_new[-1] = p_new[-2]
                p_new[:, 0] = p_new[:, 1]
                p_new[:, -1] = p_new[:, -2]

                if it > 0 and it % check_interval == 0:
                    np.subtract(p_new, p_old, out=tmp_full)
                    np.abs(tmp_full, out=tmp_full)
                    if np.max(tmp_full) < tol: return p_new, it

        else:
            for it in range(max_iter):
                if it > 0 and it % check_interval == 0: np.copyto(p_old, p_new)

                # ⚡ Bolt: Replace np.putmask and boolean mask arrays with standard strided slicing
                p_slice[0::2, 0::2] = ( (p_right[0::2, 0::2] + p_left[0::2, 0::2]) * mult_y_over_x + p_up[0::2, 0::2] + p_down[0::2, 0::2] - rhs_eff[0::2, 0::2] ) * mult_x + p_slice[0::2, 0::2] * one_minus_omega
                p_slice[1::2, 1::2] = ( (p_right[1::2, 1::2] + p_left[1::2, 1::2]) * mult_y_over_x + p_up[1::2, 1::2] + p_down[1::2, 1::2] - rhs_eff[1::2, 1::2] ) * mult_x + p_slice[1::2, 1::2] * one_minus_omega

                p_slice[0::2, 1::2] = ( (p_right[0::2, 1::2] + p_left[0::2, 1::2]) * mult_y_over_x + p_up[0::2, 1::2] + p_down[0::2, 1::2] - rhs_eff[0::2, 1::2] ) * mult_x + p_slice[0::2, 1::2] * one_minus_omega
                p_slice[1::2, 0::2] = ( (p_right[1::2, 0::2] + p_left[1::2, 0::2]) * mult_y_over_x + p_up[1::2, 0::2] + p_down[1::2, 0::2] - rhs_eff[1::2, 0::2] ) * mult_x + p_slice[1::2, 0::2] * one_minus_omega

                p_new[0] = p_new[1]
                p_new[-1] = p_new[-2]
                p_new[:, 0] = p_new[:, 1]
                p_new[:, -1] = p_new[:, -2]

                if it > 0 and it % check_interval == 0:
                    np.subtract(p_new, p_old, out=tmp_full)
                    np.abs(tmp_full, out=tmp_full)
                    if np.max(tmp_full) < tol: return p_new, it

        # Final check if loop finishes
        # Check against previous iteration which wasn't saved, so we just return.
        # It's an edge case, we can assume it hit max_iter without converging.
        return p_new, max_iter
