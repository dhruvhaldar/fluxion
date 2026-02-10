import numpy as np

def compute_divergence(u, v, grid):
    """
    Computes divergence of velocity field (u, v) at cell centers.
    u: (nx+1, ny) defined at vertical faces.
    v: (nx, ny+1) defined at horizontal faces.
    Returns: div (nx, ny)
    """
    dx, dy = grid.dx, grid.dy
    return (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy

def compute_gradient(p, grid):
    """
    Computes gradient of scalar p at cell faces.
    p: (nx, ny) defined at cell centers.
    Returns:
        grad_x: (nx+1, ny) at u-faces
        grad_y: (nx, ny+1) at v-faces
    """
    dx, dy = grid.dx, grid.dy
    grad_x = np.zeros((grid.nx+1, grid.ny))
    grad_y = np.zeros((grid.nx, grid.ny+1))

    # Interior faces
    grad_x[1:-1, :] = (p[1:, :] - p[:-1, :]) / dx
    grad_y[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dy

    # Boundaries are left as 0.0 (Homogeneous Neumann assumption common in PPE)
    return grad_x, grad_y

def compute_laplacian(phi, grid):
    """
    Computes Laplacian of phi at cell centers using central differences.
    phi: (nx, ny)
    Returns: lap (nx, ny)
    Note: Boundaries are not computed (remain 0) or should be handled by ghost cells.
    Here we compute for interior cells 1:-1.
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    lap = np.zeros_like(phi)

    # Interior
    lap[1:-1, 1:-1] = (
        (phi[2:, 1:-1] - 2*phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / dx**2 +
        (phi[1:-1, 2:] - 2*phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dy**2
    )
    return lap

def convection_term(phi, u, v, grid, scheme='central'):
    """
    Computes div(u * phi) at cell centers.
    u: (nx+1, ny)
    v: (nx, ny+1)
    phi: (nx, ny)
    scheme: 'central', 'upwind', 'quick'
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    flux_x = np.zeros((nx+1, ny))
    flux_y = np.zeros((nx, ny+1))

    # --- X-Fluxes ---
    # Faces i=1 to nx-1 are interior
    # Left cell: i-1, Right cell: i

    # Central Difference
    if scheme == 'central':
        flux_x[1:-1, :] = u[1:-1, :] * 0.5 * (phi[:-1, :] + phi[1:, :])
        flux_y[:, 1:-1] = v[:, 1:-1] * 0.5 * (phi[:, :-1] + phi[:, 1:])

    # First Order Upwind
    elif scheme == 'upwind':
        # X-direction
        u_int = u[1:-1, :]
        phi_L = phi[:-1, :]
        phi_R = phi[1:, :]
        mask_u = u_int > 0
        val_x = np.zeros_like(u_int)
        val_x[mask_u] = phi_L[mask_u]
        val_x[~mask_u] = phi_R[~mask_u]
        flux_x[1:-1, :] = u_int * val_x

        # Y-direction
        v_int = v[:, 1:-1]
        phi_D = phi[:, :-1]
        phi_U = phi[:, 1:]
        mask_v = v_int > 0
        val_y = np.zeros_like(v_int)
        val_y[mask_v] = phi_D[mask_v]
        val_y[~mask_v] = phi_U[~mask_v]
        flux_y[:, 1:-1] = v_int * val_y

    # QUICK Scheme
    elif scheme == 'quick':
        # 1D QUICK: phi_f = 1/8 * (6*phi_C + 3*phi_D - phi_U)
        # where C is immediate upstream, D is immediate downstream, U is far upstream

        # X-direction interior (i=2 to nx-2 to have enough points)
        # We fall back to CDS/Upwind for i=1 and i=nx-1

        # Ranges for vectorized QUICK
        # i corresponds to face index.
        # C (upstream) depends on u sign.

        # Let's implement full vectorized QUICK for interior faces 2:-2
        # u[i] > 0: C=i-1, D=i, U=i-2
        # u[i] < 0: C=i, D=i-1, U=i+1

        u_int = u[2:-2, :]
        mask_u = u_int > 0

        # phi indices for flux at face i (where i is index in 0..nx)
        # In python slice 2:-2 corresponds to indices 2, 3, ... nx-2.
        # Cells:
        # i-2: phi[:-3]
        # i-1: phi[1:-2]
        # i:   phi[2:-1]
        # i+1: phi[3:]

        # Slice phi arrays to align with faces 2:-2
        phi_mm = phi[:-3, :] # i-2
        phi_m  = phi[1:-2, :] # i-1
        phi_p  = phi[2:-1, :] # i
        phi_pp = phi[3:, :]   # i+1

        val_x = np.zeros_like(u_int)

        # u > 0
        # phi_f = 1/8 * (6*phi_m + 3*phi_p - phi_mm)
        val_x[mask_u] = 0.125 * (6*phi_m[mask_u] + 3*phi_p[mask_u] - phi_mm[mask_u])

        # u < 0
        # phi_f = 1/8 * (6*phi_p + 3*phi_m - phi_pp)
        val_x[~mask_u] = 0.125 * (6*phi_p[~mask_u] + 3*phi_m[~mask_u] - phi_pp[~mask_u])

        flux_x[2:-2, :] = u_int * val_x

        # Fill boundaries (1 and -2) with Upwind/Central
        # Let's use Upwind for robustness at boundaries
        # Face 1
        mask_1 = u[1,:] > 0
        flux_x[1, :][mask_1] = u[1,:][mask_1] * phi[0,:][mask_1]
        flux_x[1, :][~mask_1] = u[1,:][~mask_1] * phi[1,:][~mask_1]

        # Face nx-1
        mask_last = u[-2,:] > 0 # index -2 is second to last face
        flux_x[-2, :][mask_last] = u[-2,:][mask_last] * phi[-2,:][mask_last]
        flux_x[-2, :][~mask_last] = u[-2,:][~mask_last] * phi[-1,:][~mask_last]

        # Y-Direction (similar logic)
        v_int = v[:, 2:-2]
        mask_v = v_int > 0

        phi_yy = phi[:, :-3]
        phi_y  = phi[:, 1:-2]
        phi_Y  = phi[:, 2:-1]
        phi_YY = phi[:, 3:]

        val_y = np.zeros_like(v_int)
        val_y[mask_v] = 0.125 * (6*phi_y[mask_v] + 3*phi_Y[mask_v] - phi_yy[mask_v])
        val_y[~mask_v] = 0.125 * (6*phi_Y[~mask_v] + 3*phi_y[~mask_v] - phi_YY[~mask_v])

        flux_y[:, 2:-2] = v_int * val_y

        # Y Boundaries
        mask_1 = v[:, 1] > 0
        flux_y[:, 1][mask_1] = v[:, 1][mask_1] * phi[:, 0][mask_1]
        flux_y[:, 1][~mask_1] = v[:, 1][~mask_1] * phi[:, 1][~mask_1]

        mask_last = v[:, -2] > 0
        flux_y[:, -2][mask_last] = v[:, -2][mask_last] * phi[:, -2][mask_last]
        flux_y[:, -2][~mask_last] = v[:, -2][~mask_last] * phi[:, -1][~mask_last]

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # Boundary Fluxes (Simple approx or 0)
    # Using cell value at boundary
    flux_x[0, :] = u[0, :] * phi[0, :]
    flux_x[-1, :] = u[-1, :] * phi[-1, :]
    flux_y[:, 0] = v[:, 0] * phi[:, 0]
    flux_y[:, -1] = v[:, -1] * phi[:, -1]

    # Compute Divergence
    conv = (flux_x[1:, :] - flux_x[:-1, :]) / dx + (flux_y[:, 1:] - flux_y[:, :-1]) / dy

    return conv
