"""Motion and measurement models used by the simulator and the baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearGaussianModel:
    """x_{k+1} = F x_k + w,  w ~ N(0, Q);  z_k = H x_k + v,  v ~ N(0, R)."""

    F: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: np.ndarray


def constant_velocity_model(
    dt: float = 1.0, process_noise: float = 0.1, measurement_noise: float = 5.0
) -> LinearGaussianModel:
    """2D constant-velocity (CV) model with state [x, y, vx, vy] and position measurements.

    ``process_noise`` is the white-acceleration spectral density q; ``measurement_noise`` is the
    standard deviation of each position measurement component.
    """
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    q1 = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], dtype=np.float64) * process_noise
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = q1
    Q[np.ix_([1, 3], [1, 3])] = q1
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    R = np.eye(2) * measurement_noise**2
    return LinearGaussianModel(F=F, Q=Q, H=H, R=R)
