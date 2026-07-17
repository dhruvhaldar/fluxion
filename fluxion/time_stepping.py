import numpy as np

def euler_step(field, rhs, dt):
    """
    Performs a single Explicit Euler time step.
    new_field = field + dt * rhs
    """
    return field + dt * rhs

def rk4_step(field, rhs_func, dt, *args, **kwargs):
    """
    Performs a single Runge-Kutta 4th Order time step.
    rhs_func: Callable that returns the derivative (RHS) of the field.
              Signature: rhs_func(field, *args, **kwargs)
    """
    # ⚡ Bolt: Using standard float multiplication is faster than relying on integer division or integer casting during math
    half_dt = 0.5 * dt
    k1 = rhs_func(field, *args, **kwargs)
    k2 = rhs_func(field + half_dt * k1, *args, **kwargs)
    k3 = rhs_func(field + half_dt * k2, *args, **kwargs)
    k4 = rhs_func(field + dt * k3, *args, **kwargs)

    # ⚡ Bolt: Factoring out operations saves two array allocations per RK4 step natively in the C backend
    return field + (dt / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)
