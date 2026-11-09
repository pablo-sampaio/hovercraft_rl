import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hovercraft_rl.envs import HovercraftTrackEnv, make_env
from hovercraft_rl.track.track import Track


def square():
    return Track([[0, 0], [100, 0], [100, 100], [0, 100]], half_width=4)


def observe(name, params=None, track=None, state=None):
    env = HovercraftTrackEnv(track=track or square(), observation=name, observation_params=params)
    if state is None:
        state = {"x": 20.0, "y": 2.0, "psi": np.pi / 2, "u": 3.0, "v": -1.0, "r": 0.1}
    obs, info = env.reset(options={"state": state})
    return env, obs, info


def test_observations_represent_same_geometry():
    params = {"lookahead_distances": [5, 15, 25], "include_directions": True}
    world, w, info = observe("world_frame", params)
    relative, h, relative_info = observe("hovercraft_frame", params)

    np.testing.assert_allclose(w, [0, 2, .5, 0, 1, 1, 0, 3, -1, .1,
                                   5, 0, 15, 0, 25, 0, 1, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(h, [2, 0, .5, 0, -1, 3, -1, .1,
                                   0, -5, 0, -15, 0, -25, 0, -1, 0, -1], atol=1e-6)
    np.testing.assert_allclose(info["lookahead_points"], relative_info["lookahead_points"])
    assert len(world.observation_builder.names(world)) == len(w)
    assert len(relative.observation_builder.names(relative)) == len(h)
    assert world.observation_space.shape == w.shape
    assert relative.observation_space.shape == h.shape


def test_rotation_invariance_and_constant_heading():
    params = {"n_lookahead": 2, "lookahead_spacing": 5, "include_hovercraft_heading": True}
    state = {"x": 20, "y": 2, "psi": np.pi / 2}
    _, first, _ = observe("hovercraft_frame", params, state=state)
    rotated = Track([[0, 0], [0, 100], [-100, 100], [-100, 0]], half_width=4)
    _, second, _ = observe("hovercraft_frame", params, rotated,
                           {"x": -2, "y": 20, "psi": np.pi})
    np.testing.assert_allclose(first, second, atol=1e-6)
    np.testing.assert_allclose(first[3:5], [1, 0], atol=1e-6)
    _, world, _ = observe("world_frame", params, state=state)
    assert len(first) == len(world)


def test_defaults_and_reference_point_are_shared():
    for name, length in [("world_frame", 20), ("hovercraft_frame", 18)]:
        env, obs, _ = observe(name)
        assert len(obs) == length
        np.testing.assert_array_equal(env.observation_builder.distances, [5, 10, 15, 20, 25])


@pytest.mark.parametrize("name", ["world_frame", "hovercraft_frame"])
@pytest.mark.parametrize("extra", [
    {"lookahead_distances": []},
    {"lookahead_distances": [5, 15, 25], "include_directions": True,
     "include_hovercraft_heading": True},
])
def test_optional_lateral_error_only_removes_that_signal(name, extra):
    with_error, full, _ = observe(name, extra)
    explicit, explicit_obs, _ = observe(name, {**extra, "include_lateral_error": True})
    without_error, reduced, _ = observe(name, {**extra, "include_lateral_error": False})
    names = with_error.observation_builder.names(with_error)
    reduced_names = without_error.observation_builder.names(without_error)
    index = names.index("e_lat_norm")
    assert full[index] == pytest.approx(0.5)
    np.testing.assert_array_equal(explicit_obs, full)
    np.testing.assert_array_equal(reduced, np.delete(full, index))
    assert reduced_names == names[:index] + names[index + 1:]
    assert without_error.observation_space.shape == reduced.shape
    assert without_error.observation_space.contains(reduced)
    for env in (with_error, explicit, without_error):
        env.close()


def test_directions_at_a_corner_use_chords_not_tangents():
    env, obs, _ = observe("world_frame", {"lookahead_distances": [70, 90, 110],
                                          "include_directions": True})
    np.testing.assert_allclose(obs[10:16], [70, 0, 80, 10, 80, 30], atol=1e-6)
    np.testing.assert_allclose(obs[16:], [2**-0.5, 2**-0.5, 0, 1], atol=1e-6)


@pytest.mark.parametrize("name", ["world_frame", "hovercraft_frame"])
@pytest.mark.parametrize("params", [
    {},
    {"lookahead_distances": []},
    {"n_lookahead": 1, "include_directions": True},
    {"lookahead_distances": [2, 3, 10], "include_directions": True},
])
def test_shapes_and_gym_contract(name, params):
    env = HovercraftTrackEnv(track=square(), observation=name, observation_params=params)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset()
    assert len(obs) == len(env.observation_builder.names(env))
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize("params", [
    {"lookahead_distances": [2, 2]},
    {"lookahead_distances": [2, 1]},
    {"lookahead_distances": [0]},
    {"lookahead_distances": [float("nan")]},
    {"lookahead_distances": [1], "n_lookahead": 1},
    {"lookahead_distances": [400]},
    {"n_lookahead": 4, "lookahead_spacing": 100},
    {"n_lookahead": -1},
    {"lookahead_spacing": 0},
])
def test_invalid_lookahead(params):
    with pytest.raises(ValueError):
        HovercraftTrackEnv(track=square(), observation="world_frame", observation_params=params)


def test_coincident_points_warn_and_use_zero_direction():
    crossing = Track([[0, 0], [2, 2], [0, 2], [2, 0]], half_width=1)
    first_s = np.sqrt(2)
    second_s = 2 * np.sqrt(2) + 2 + np.sqrt(2)
    env = HovercraftTrackEnv(track=crossing, observation="world_frame",
                             observation_params={"lookahead_distances": [first_s, second_s],
                                                 "include_directions": True})
    with pytest.warns(RuntimeWarning, match="coincidem"):
        obs, _ = env.reset()
    np.testing.assert_array_equal(obs[-2:], [0, 0])
    assert np.isfinite(obs).all()


def test_shorter_evaluation_track_is_validated():
    with pytest.raises(ValueError, match="comprimento da pista"):
        make_env(Track([[0, 0], [2, 0], [2, 2], [0, 2]], half_width=1),
                 observation="world_frame", observation_params={"lookahead_distances": [8]})
