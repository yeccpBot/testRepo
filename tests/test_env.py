import numpy as np
from gymnasium.utils.env_checker import check_env

from trackrl.envs import SingleTargetTrackingEnv


def zero(obs):
    return np.zeros(4, dtype=np.float32)


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        done = terminated or truncated
    return total


def test_gymnasium_api():
    check_env(SingleTargetTrackingEnv(max_steps=10), skip_render_check=True)


def test_deterministic_with_seed():
    env = SingleTargetTrackingEnv(max_steps=20)
    assert run_episode(env, zero, seed=3) == run_episode(env, zero, seed=3)


def test_alpha_beta_policy_beats_pure_prediction():
    """A hand-written alpha-beta update must score better than ignoring measurements."""
    env = SingleTargetTrackingEnv(max_steps=100)
    alpha, beta = 0.5, 0.2

    def alpha_beta(obs):
        innov = obs[:2] * env.meas_std
        correction = np.concatenate([alpha * innov, beta * innov])  # dt = 1
        return np.clip(correction / env.action_scale, -1, 1).astype(np.float32)

    seeds = range(10)
    ab = np.mean([run_episode(env, alpha_beta, s) for s in seeds])
    pred_only = np.mean([run_episode(env, zero, s) for s in seeds])
    assert ab > pred_only
    # Mean per-step position error of the alpha-beta tracker is on the order of the measurement std.
    assert -ab / 100 < 2.0
