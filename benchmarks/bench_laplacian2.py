import numpy as np
import time

nx, ny = 200, 200
phi = np.random.rand(nx, ny)
lap1 = np.zeros_like(phi)
lap2 = np.zeros_like(phi)
inv_dx2 = 1.0 / 0.01

start = time.time()
for _ in range(5000):
    lap1[1:-1, 1:-1] = (
        phi[2:, 1:-1] + phi[:-2, 1:-1] + phi[1:-1, 2:] + phi[1:-1, :-2] - 4*phi[1:-1, 1:-1]
    ) * inv_dx2
end1 = time.time()
t1 = end1 - start

tmp1 = np.zeros_like(phi[1:-1, 1:-1])

start = time.time()
for _ in range(5000):
    lap_int = lap2[1:-1, 1:-1]
    phi_int = phi[1:-1, 1:-1]

    np.add(phi[2:, 1:-1], phi[:-2, 1:-1], out=lap_int)
    np.add(lap_int, phi[1:-1, 2:], out=lap_int)
    np.add(lap_int, phi[1:-1, :-2], out=lap_int)
    np.multiply(phi_int, 4.0, out=tmp1)
    np.subtract(lap_int, tmp1, out=lap_int)
    np.multiply(lap_int, inv_dx2, out=lap_int)
end2 = time.time()
t2 = end2 - start

print(f"Uniform Original: {t1:.4f} seconds")
print(f"Uniform Optimized: {t2:.4f} seconds")

lap1 = np.zeros_like(phi)
lap2 = np.zeros_like(phi)
inv_dy2 = 1.0 / 0.02

start = time.time()
for _ in range(5000):
    lap1[1:-1, 1:-1] = (
        (phi[2:, 1:-1] - 2*phi[1:-1, 1:-1] + phi[:-2, 1:-1]) * inv_dx2 +
        (phi[1:-1, 2:] - 2*phi[1:-1, 1:-1] + phi[1:-1, :-2]) * inv_dy2
    )
end1 = time.time()
t1 = end1 - start

tmp2 = np.zeros_like(phi[1:-1, 1:-1])
start = time.time()
for _ in range(5000):
    lap_int = lap2[1:-1, 1:-1]
    phi_int = phi[1:-1, 1:-1]

    np.add(phi[2:, 1:-1], phi[:-2, 1:-1], out=tmp1)
    np.multiply(phi_int, 2.0, out=tmp2)
    np.subtract(tmp1, tmp2, out=tmp1)
    np.multiply(tmp1, inv_dx2, out=tmp1)

    np.add(phi[1:-1, 2:], phi[1:-1, :-2], out=lap_int)
    np.subtract(lap_int, tmp2, out=lap_int)
    np.multiply(lap_int, inv_dy2, out=lap_int)

    np.add(lap_int, tmp1, out=lap_int)
end2 = time.time()
t2 = end2 - start

print(f"Non-Uniform Original: {t1:.4f} seconds")
print(f"Non-Uniform Optimized: {t2:.4f} seconds")
