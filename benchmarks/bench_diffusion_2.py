import numpy as np
import time

nx, ny = 200, 200
u_interior = np.random.rand(nx, ny)
inv_dy2 = 0.5
inv_2dy = 0.5

d2u_dy2 = np.empty_like(u_interior)
du_dy = np.empty_like(u_interior)

start = time.time()
for _ in range(5000):
    d2u_dy2[:, 1:-1] = (u_interior[:, 2:] - 2*u_interior[:, 1:-1] + u_interior[:, :-2]) * inv_dy2
    du_dy[:, 1:-1] = (u_interior[:, 2:] - u_interior[:, :-2]) * inv_2dy
end = time.time()
print(f"Original Time: {end - start:.4f} seconds")

start = time.time()
for _ in range(5000):
    np.multiply(u_interior[:, 1:-1], 2.0, out=d2u_dy2[:, 1:-1])
    np.subtract(u_interior[:, 2:], d2u_dy2[:, 1:-1], out=d2u_dy2[:, 1:-1])
    np.add(d2u_dy2[:, 1:-1], u_interior[:, :-2], out=d2u_dy2[:, 1:-1])
    np.multiply(d2u_dy2[:, 1:-1], inv_dy2, out=d2u_dy2[:, 1:-1])

    np.subtract(u_interior[:, 2:], u_interior[:, :-2], out=du_dy[:, 1:-1])
    np.multiply(du_dy[:, 1:-1], inv_2dy, out=du_dy[:, 1:-1])
end = time.time()
print(f"Optimized Time: {end - start:.4f} seconds")
