import numpy as np
import pytest

from hovercraft_rl.physics import Hovercraft3DOFDrag, Hovercraft3DOFParams, make_physics


def test_euler_step_matches_three_damping_equations():
    params = Hovercraft3DOFParams(beta_v=0.2, beta_u=0.1, beta_r=0.3)
    model = Hovercraft3DOFDrag(params)
    state = np.array([1.0, 2.0, 0.3, 1.5, -0.4, 0.2])
    model.reset(state)
    action = np.array([0.7, -0.3])
    next_state = model.step(action)
    x, y, psi, u, v, r = state
    h = params.dt
    expected = state + h * np.array([
        u * np.cos(psi) - v * np.sin(psi),
        u * np.sin(psi) + v * np.cos(psi),
        r,
        v * r + action[0] - params.beta_u * u,
        -u * r - params.beta_v * v,
        action[1] - params.beta_r * r,
    ])
    np.testing.assert_allclose(next_state, expected)
    assert model.t == pytest.approx(params.dt)


def test_article_parameters_reproduce_article_model():
    model = Hovercraft3DOFDrag(Hovercraft3DOFParams(beta_v=0.2, beta_u=0.0, beta_r=0.0))
    state = np.array([1.0, 2.0, 0.3, 1.5, -0.4, 0.2])
    model.reset(state)
    action = np.array([0.7, -0.3])
    next_state = model.step(action)
    x, y, psi, u, v, r = state
    expected = state + 0.02 * np.array([
        u * np.cos(psi) - v * np.sin(psi),
        u * np.sin(psi) + v * np.cos(psi),
        r,
        v * r + action[0],
        -u * r - 0.2 * v,
        action[1],
    ])
    np.testing.assert_allclose(next_state, expected)


def test_action_is_saturated_and_state_is_not():
    model = Hovercraft3DOFDrag()
    model.reset({"u": 50.0})
    state = model.step([10.0, -10.0])  # saturates at (2, -1)
    assert state[3] == pytest.approx(50.0 + 0.02 * 2.0)
    assert state[5] == pytest.approx(0.02 * -1.0)


def test_no_damping_parameters_preserve_surge_and_yaw():
    """With zero beta_u/beta_r, zero input and sway preserve u and r as in the paper."""
    model = Hovercraft3DOFDrag()
    model.reset({"u": 3.0})
    for _ in range(500):
        model.step([0.0, 0.0])
    assert model.state_dict()["u"] == pytest.approx(3.0)


def test_constant_inputs_from_rest_with_rk4():
    """With beta_r=0, constant yaw input gives r=tau_r*t and psi=tau_r*t²/2."""
    model = Hovercraft3DOFDrag(Hovercraft3DOFParams(integrator="rk4"))
    model.reset()
    n = int(round(10.5 / model.dt))
    for _ in range(n):
        state = model.step([0.8, 0.2])
    t = n * model.dt
    assert state[5] == pytest.approx(0.2 * t)
    assert state[2] == pytest.approx(0.5 * 0.2 * t**2)


def test_energy_balance_includes_surge_and_sway_drag():
    """d/dt (u²+v²)/2 = u*tau_u - beta_u*u² - beta_v*v²."""
    params = Hovercraft3DOFParams(beta_v=0.2, beta_u=0.1, integrator="rk4", dt=1e-3)
    model = Hovercraft3DOFDrag(params)
    model.reset({"u": 2.0, "v": 1.0, "r": 0.3})
    energy_before = 0.5 * (2.0**2 + 1.0**2)
    state = model.step([0.5, 0.0])
    energy_after = 0.5 * (state[3] ** 2 + state[4] ** 2)
    expected_rate = 2.0 * 0.5 - 0.1 * 2.0**2 - 0.2 * 1.0**2
    assert (energy_after - energy_before) / params.dt == pytest.approx(expected_rate, rel=1e-2)


def test_reset_partial_dict_and_registry():
    model = make_physics("3dof+drag", {"beta_v": 0.5, "dt": 0.01})
    assert isinstance(model, Hovercraft3DOFDrag)
    assert model.dt == 0.01 and model.params.beta_v == 0.5
    model.reset({"x": 3.0})
    assert model.state_dict() == {"x": 3.0, "y": 0.0, "psi": 0.0, "u": 0.0, "v": 0.0, "r": 0.0}
    with pytest.raises(KeyError):
        model.reset({"z": 1.0})
    with pytest.raises(KeyError):
        make_physics("hovercraft3dof")
    with pytest.raises(KeyError):
        make_physics("hovercraft3dof_drag")


def test_drag_gives_terminal_surge_speed():
    model = make_physics("3dof+drag", {"beta_u": 0.1, "beta_r": 1.0})
    model.reset()
    for _ in range(int(100 / model.dt)):
        state = model.step([2.0, 0.0])
    assert state[3] == pytest.approx(2.0 / 0.1, rel=1e-3)
