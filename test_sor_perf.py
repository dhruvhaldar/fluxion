import numpy as np
import time

p_slice = np.random.rand(500, 500)
p_right = np.random.rand(500, 500)
p_left = np.random.rand(500, 500)

def slow():
    start = time.time()
    for _ in range(5000):
        p_gs = np.empty_like(p_slice)
        np.add(p_right, p_left, out=p_gs)
    return time.time() - start

def fast():
    start = time.time()
    for _ in range(5000):
        p_gs = np.empty(p_slice.shape)
        np.add(p_right, p_left, out=p_gs)
    return time.time() - start

print(f"slow: {slow():.4f}")
print(f"fast: {fast():.4f}")
