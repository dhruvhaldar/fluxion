import numpy as np

class StaggeredGrid:
    """
    Represents a 2D structured staggered grid for Finite Volume Method.
    Variables:
    - p (pressure, scalars): Cell centers
    - u (x-velocity): Vertical cell faces
    - v (y-velocity): Horizontal cell faces
    """
    def __init__(self, nx, ny, lx=1.0, ly=1.0):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / nx
        self.dy = ly / ny

        # Coordinate arrays
        # Cell centers (for scalars like pressure)
        self.x_c = np.linspace(self.dx/2, self.lx - self.dx/2, nx)
        self.y_c = np.linspace(self.dy/2, self.ly - self.dy/2, ny)

        # Meshgrids for vectorized operations
        # Indexing 'ij' ensures matrix indexing (x corresponds to rows/first dim, y to cols/second dim)
        # Wait, usually meshgrid 'ij' means:
        # X[i, j] depends on x[i]
        # Y[i, j] depends on y[j]
        self.X_c, self.Y_c = np.meshgrid(self.x_c, self.y_c, indexing='ij')

        # u-velocity locations (staggered in x)
        # Dimensions: (nx+1, ny)
        self.x_u = np.linspace(0, self.lx, nx + 1)
        self.y_u = self.y_c
        self.X_u, self.Y_u = np.meshgrid(self.x_u, self.y_u, indexing='ij')

        # v-velocity locations (staggered in y)
        # Dimensions: (nx, ny+1)
        self.x_v = self.x_c
        self.y_v = np.linspace(0, self.ly, ny + 1)
        self.X_v, self.Y_v = np.meshgrid(self.x_v, self.y_v, indexing='ij')

    def initialize_field(self, func, location='center'):
        """
        Initialize a field based on a function f(x, y).
        """
        if location == 'center':
            return func(self.X_c, self.Y_c)
        elif location == 'u':
            return func(self.X_u, self.Y_u)
        elif location == 'v':
            return func(self.X_v, self.Y_v)
        else:
            raise ValueError("Location must be 'center', 'u', or 'v'")
