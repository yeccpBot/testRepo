import numpy as np

from trackrl import KalmanFilter, constant_velocity_model


def test_kalman_beats_raw_measurements():
    rng = np.random.default_rng(0)
    model = constant_velocity_model(dt=1.0, process_noise=0.1, measurement_noise=5.0)
    x = np.array([0.0, 0.0, 10.0, -5.0])
    kf = KalmanFilter(model, x0=np.zeros(4), P0=np.diag([100.0, 100.0, 400.0, 400.0]))

    meas_err, kf_err = [], []
    for k in range(200):
        x = model.F @ x + rng.multivariate_normal(np.zeros(4), model.Q)
        z = model.H @ x + rng.multivariate_normal(np.zeros(2), model.R)
        est = kf.step(z)
        if k >= 20:  # skip convergence transient
            meas_err.append(np.linalg.norm(z - x[:2]))
            kf_err.append(np.linalg.norm(est[:2] - x[:2]))

    assert np.mean(kf_err) < 0.8 * np.mean(meas_err)
    # Covariance stays symmetric positive definite.
    assert np.allclose(kf.P, kf.P.T)
    assert np.all(np.linalg.eigvalsh(kf.P) > 0)
