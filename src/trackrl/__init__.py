"""trackrl: end-to-end neural track processing trained with reinforcement learning."""

from trackrl.kalman import KalmanFilter
from trackrl.models import constant_velocity_model

__all__ = ["KalmanFilter", "constant_velocity_model"]
__version__ = "0.1.0"
