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
    k1 = rhs_func(field, *args, **kwargs)
    k2 = rhs_func(field + 0.5 * dt * k1, *args, **kwargs)
    k3 = rhs_func(field + 0.5 * dt * k2, *args, **kwargs)
    k4 = rhs_func(field + dt * k3, *args, **kwargs)

    return field + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
