import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import hovercraft_rl  # noqa: F401  (registra o ambiente)
from hovercraft_rl.envs import (BodyFrameObservation, CompositeReward, HovercraftTrackEnv,
                                LateralErrorPenalty, ProgressReward, ReferenceErrorObservation, make_env)


@pytest.mark.parametrize("obs", [ReferenceErrorObservation(), BodyFrameObservation(),
                                 ReferenceErrorObservation(neighbor_headings=False, n_lookahead=0)])
def test_check_env(obs):
    check_env(HovercraftTrackEnv("oval", observation=obs), skip_render_check=True)


def test_reference_error_observation_layout():
    env = HovercraftTrackEnv("oval", observation=ReferenceErrorObservation(n_lookahead=3))
    obs, info = env.reset()
    assert obs.shape == (7 + 4 + 6,)
    np.testing.assert_allclose(obs[:2], 0.0, atol=1e-9)        # largada sobre a linha base
    np.testing.assert_allclose(obs[2:4], [1.0, 0.0], atol=1e-6)  # oval começa apontando para +x


def test_gym_make_has_time_limit():
    env = gym.make("HovercraftTrack-v0", track="circle")
    assert env.spec.max_episode_steps == 5000


def test_truncation_and_progress():
    env = make_env("oval", max_episode_steps=50)
    env.reset(seed=0)
    total = 0.0
    for _ in range(50):
        _, r, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        total += r
    assert trunc and not term
    assert info["progress"] > 0 and total == pytest.approx(info["progress"])


def test_terminates_off_track_with_penalty():
    env = make_env("circle")
    env.reset(seed=0)
    for _ in range(5000):   # reto, a pista curva => sai
        _, r, term, trunc, info = env.step(np.array([2.0, 0.0], dtype=np.float32))
        if term:
            break
    assert term and info["off_track"]
    assert info["reward_terms"]["off_track"] == pytest.approx(-10.0)


def test_random_start_and_custom_reward():
    rew = CompositeReward([(ProgressReward(), 1.0), (LateralErrorPenalty(), 0.5)])
    env = HovercraftTrackEnv("gp", start="random", reward=rew)
    s0 = {env.reset(seed=i)[1]["s"] for i in range(5)}
    assert len(s0) == 5
    _, _, _, _, info = env.step(env.action_space.sample())
    assert set(info["reward_terms"]) == {"progress", "lateral_error"}


@pytest.mark.parametrize("view", ["global", "ego"])
def test_rgb_render(view):
    env = HovercraftTrackEnv("gp", render_mode="rgb_array", render_view=view)
    env.reset()
    env.step(np.array([2.0, 0.5], dtype=np.float32))
    frame = env.render()
    assert frame.shape == (720, 960, 3) and frame.dtype == np.uint8
    env.close()


@pytest.mark.parametrize("mode", ["discrete", "multidiscrete"])
def test_discrete_action_variants(mode):
    env = make_env("oval", action_mode=mode, tau_u_levels=3, tau_r_levels=[-1.0, -0.3, 0.0, 0.3, 1.0])
    check_env(env, skip_render_check=True)
    env.reset(seed=0)
    if mode == "discrete":
        assert env.action_space.n == 15
        a = 14                                  # último = (tau_u máx, tau_r máx)
    else:
        assert list(env.action_space.nvec) == [3, 5]
        a = np.array([2, 4])
    env.step(a)
    np.testing.assert_allclose(env.unwrapped.last_action, [2.0, 1.0])


def test_gym_make_discrete():
    env = gym.make("HovercraftTrackDiscrete-v0", track="circle")
    assert env.action_space.n == 9 and env.spec.max_episode_steps == 5000
    env.reset(seed=0)
    env.step(4)
