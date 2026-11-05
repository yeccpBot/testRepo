"""Single-target 2D tracking environment (phase-0 sandbox, see docs/rl-track-processing-approaches.md).

Each step the agent receives a noisy position measurement and the constant-velocity prediction of
its previous estimate. The action is a correction added to that prediction, so the policy plays the
role of a learned filter update. The reward is the negative normalized position error.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trackrl.models import LinearGaussianModel, constant_velocity_model


class SingleTargetTrackingEnv(gym.Env):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        model: LinearGaussianModel | None = None,
        max_steps: int = 100,
        initial_speed: float = 10.0,
        position_range: float = 1000.0,
    ) -> None:
        super().__init__()
        self.model = model or constant_velocity_model()
        self.max_steps = max_steps
        self.initial_speed = initial_speed
        self.position_range = position_range

        self.meas_std = float(np.sqrt(self.model.R[0, 0]))
        # Normalization: positions in units of measurement std, velocities in units of initial speed.
        self.state_scale = np.array(
            [self.meas_std, self.meas_std, initial_speed, initial_speed], dtype=np.float64
        )
        # The largest correction the agent may apply per step, per state component.
        self.action_scale = 3.0 * self.state_scale

        # Observation: [innovation (2), predicted velocity (2)], normalized. Absolute positions are
        # omitted on purpose so the policy is translation invariant.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self._x_true = np.zeros(4)
        self._x_est = np.zeros(4)
        self._x_pred = np.zeros(4)
        self._z = np.zeros(2)
        self._t = 0

    # --- simulation -----------------------------------------------------------------------------
    def _measure(self, x: np.ndarray) -> np.ndarray:
        noise = self.np_random.multivariate_normal(np.zeros(2), self.model.R)
        return self.model.H @ x + noise

    def _propagate(self, x: np.ndarray) -> np.ndarray:
        noise = self.np_random.multivariate_normal(np.zeros(4), self.model.Q)
        return self.model.F @ x + noise

    def _obs(self) -> np.ndarray:
        innovation = (self._z - self.model.H @ self._x_pred) / self.meas_std
        velocity = self._x_pred[2:] / self.initial_speed
        return np.concatenate([innovation, velocity]).astype(np.float32)

    # --- gym API --------------------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        pos = self.np_random.uniform(-self.position_range, self.position_range, size=2)
        heading = self.np_random.uniform(0.0, 2 * np.pi)
        vel = self.initial_speed * np.array([np.cos(heading), np.sin(heading)])
        self._x_true = np.concatenate([pos, vel])
        self._t = 0

        # Track initiation from the first measurement, velocity unknown.
        z0 = self._measure(self._x_true)
        self._x_est = np.array([z0[0], z0[1], 0.0, 0.0])

        self._x_true = self._propagate(self._x_true)
        self._x_pred = self.model.F @ self._x_est
        self._z = self._measure(self._x_true)
        return self._obs(), {"x_true": self._x_true.copy()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._x_est = self._x_pred + action * self.action_scale

        pos_err = self._x_est[:2] - self._x_true[:2]
        reward = -float(np.linalg.norm(pos_err) / self.meas_std)
        info = {"x_true": self._x_true.copy(), "x_est": self._x_est.copy()}

        self._t += 1
        truncated = self._t >= self.max_steps

        self._x_true = self._propagate(self._x_true)
        self._x_pred = self.model.F @ self._x_est
        self._z = self._measure(self._x_true)
        return self._obs(), reward, False, truncated, info
