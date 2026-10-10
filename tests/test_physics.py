import numpy as np
import pytest

from hovercraft_rl.physics import Hovercraft3DOF, Hovercraft3DOFParams, make_physics


def test_euler_step_matches_equations():
    m = Hovercraft3DOF()
    s0 = np.array([1.0, 2.0, 0.3, 1.5, -0.4, 0.2])
    m.reset(s0)
    a = np.array([0.7, -0.3])
    s1 = m.step(a)
    x, y, psi, u, v, r = s0
    h, beta = 0.02, 0.2
    expected = s0 + h * np.array([
        u * np.cos(psi) - v * np.sin(psi),
        u * np.sin(psi) + v * np.cos(psi),
        r,
        v * r + a[0],
        -u * r - beta * v,
        a[1],
    ])
    np.testing.assert_allclose(s1, expected)
    assert m.t == pytest.approx(0.02)


def test_action_is_saturated_and_state_is_not():
    m = Hovercraft3DOF()
    m.reset({"u": 50.0})
    s = m.step([10.0, -10.0])  # satura em (2, -1)
    assert s[3] == pytest.approx(50.0 + 0.02 * 2.0)
    assert s[5] == pytest.approx(0.02 * -1.0)


def test_no_surge_or_yaw_damping():
    """Como no artigo: sem atuação e sem sway, u e r se conservam."""
    m = Hovercraft3DOF()
    m.reset({"u": 3.0})
    for _ in range(500):
        m.step([0.0, 0.0])
    assert m.state_dict()["u"] == pytest.approx(3.0)


def test_constant_inputs_from_rest():
    """tau constante a partir do repouso: r = tau_r t, psi = tau_r t²/2 (exato com RK4)."""
    m = Hovercraft3DOF(Hovercraft3DOFParams(integrator="rk4"))
    m.reset()
    n = int(round(10.5 / 0.02))
    for _ in range(n):
        s = m.step([0.8, 0.2])
    t = n * 0.02
    assert s[5] == pytest.approx(0.2 * t)
    assert s[2] == pytest.approx(0.5 * 0.2 * t**2)


def test_energy_balance():
    """d/dt (u²+v²)/2 = u tau_u - beta v²  (só o sway dissipa)."""
    m = Hovercraft3DOF(Hovercraft3DOFParams(integrator="rk4", dt=1e-3))
    m.reset({"u": 2.0, "v": 1.0, "r": 0.3})
    e0 = 0.5 * (2.0**2 + 1.0**2)
    s = m.step([0.5, 0.0])
    e1 = 0.5 * (s[3] ** 2 + s[4] ** 2)
    assert (e1 - e0) / 1e-3 == pytest.approx(2.0 * 0.5 - 0.2 * 1.0**2, rel=1e-2)


def test_reset_partial_dict_and_registry():
    m = make_physics("hovercraft3dof", {"beta": 0.5, "dt": 0.01})
    assert m.dt == 0.01 and m.params.beta == 0.5
    m.reset({"x": 3.0})
    assert m.state_dict() == {"x": 3.0, "y": 0.0, "psi": 0.0, "u": 0.0, "v": 0.0, "r": 0.0}
    with pytest.raises(KeyError):
        m.reset({"z": 1.0})


def test_drag_variant():
    from hovercraft_rl.physics import Hovercraft3DOFDrag
    plain = Hovercraft3DOF()
    same = Hovercraft3DOFDrag()                       # beta_u = beta_r = 0 -> idêntico
    s0 = [1.0, 2.0, 0.3, 1.5, -0.4, 0.2]
    plain.reset(s0); same.reset(s0)
    np.testing.assert_allclose(plain.step([0.7, -0.3]), same.step([0.7, -0.3]))
    m = make_physics("hovercraft3dof_drag", {"beta_u": 0.1, "beta_r": 1.0})
    m.reset()
    for _ in range(int(100 / m.dt)):                  # 100 s em linha reta com propulsão máxima
        s = m.step([2.0, 0.0])
    assert s[3] == pytest.approx(2.0 / 0.1, rel=1e-3)  # velocidade terminal = tau_u / beta_u
