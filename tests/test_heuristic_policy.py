import numpy as np

from examples.run_policy import HeuristicPolicy
from hovercraft_rl.envs import make_env


def test_heuristic_policy_uses_body_frame_observation():
    env = make_env("oval", observation="body_frame")
    policy = HeuristicPolicy(env)
    obs = np.array([
        0.0, 1.0, 0.0, 2.0, 0.0, 0.0,
        5.0, 0.5, 10.0, 1.0, 15.0, 1.5, 20.0, 2.0, 25.0, 2.5,
    ], dtype=np.float32)

    action = policy(obs)
    changed_obs = obs.copy()
    changed_obs[5] = 0.1
    changed_action = policy(changed_obs)

    assert action.shape == (2,)
    assert np.isfinite(action).all()
    np.testing.assert_allclose(
        changed_action[1], action[1] - policy.k_d * 0.1, atol=1e-6
    )
    env.close()
