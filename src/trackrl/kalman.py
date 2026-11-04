"""Classical Kalman filter, the baseline every learned tracker is compared against."""

from __future__ import annotations

import numpy as np

from trackrl.models import LinearGaussianModel


class KalmanFilter:
    def __init__(self, model: LinearGaussianModel, x0: np.ndarray, P0: np.ndarray) -> None:
        self.model = model
        self.x = np.asarray(x0, dtype=np.float64).copy()
        self.P = np.asarray(P0, dtype=np.float64).copy()

    def predict(self) -> np.ndarray:
        m = self.model
        self.x = m.F @ self.x
        self.P = m.F @ self.P @ m.F.T + m.Q
        return self.x

    def update(self, z: np.ndarray) -> np.ndarray:
        m = self.model
        S = m.H @ self.P @ m.H.T + m.R
        K = np.linalg.solve(S, m.H @ self.P).T  # P H^T S^-1 (S and P are symmetric)
        self.x = self.x + K @ (np.asarray(z, dtype=np.float64) - m.H @ self.x)
        I_KH = np.eye(len(self.x)) - K @ m.H
        self.P = I_KH @ self.P @ I_KH.T + K @ m.R @ K.T  # Joseph form
        return self.x

    def step(self, z: np.ndarray) -> np.ndarray:
        self.predict()
        return self.update(z)
