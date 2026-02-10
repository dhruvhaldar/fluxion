import matplotlib.pyplot as plt
import numpy as np

def save_plot(fig, filename):
    """
    Saves a matplotlib figure to a file.
    """
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)

def plot_scalar_field(grid, phi, title="Scalar Field", filename=None):
    """
    Plots a scalar field contour.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # Transpose because meshgrid ij indexing vs plotting xy expectation
    c = ax.contourf(grid.X_c.T, grid.Y_c.T, phi.T, levels=20, cmap='viridis')
    fig.colorbar(c)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if filename:
        save_plot(fig, filename)
    else:
        plt.show()

def plot_velocity_magnitude(grid, u, v, title="Velocity Magnitude", filename=None):
    """
    Plots velocity magnitude contours.
    Interpolates u and v to cell centers.
    """
    u_c = 0.5 * (u[:-1, :] + u[1:, :])
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])
    speed = np.sqrt(u_c**2 + v_c**2)

    plot_scalar_field(grid, speed, title, filename)
