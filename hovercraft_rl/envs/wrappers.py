"""Wrappers de ação: variantes discretizadas do espaço contínuo ``[tau_u, tau_r]``."""

from __future__ import annotations

import itertools
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _levels(spec: int | Sequence[float], low: float, high: float) -> np.ndarray:
    """``int`` = n níveis igualmente espaçados em [low, high]; sequência = valores explícitos."""
    if isinstance(spec, (int, np.integer)):
        if spec < 2:
            raise ValueError("são necessários pelo menos 2 níveis por componente")
        return np.linspace(low, high, int(spec))
    vals = np.asarray(spec, dtype=np.float64)
    if np.any(vals < low) or np.any(vals > high):
        raise ValueError(f"níveis {vals} fora de [{low}, {high}]")
    return vals


class DiscreteActionWrapper(gym.ActionWrapper):
    """Ação ``Discrete(n_u * n_r)``: cada índice é uma combinação (tau_u, tau_r) de uma grade.

    Parâmetros
    ----------
    tau_u_levels, tau_r_levels : nº de níveis (igualmente espaçados nos limites da física) ou
        lista explícita de valores. Padrão: 3 × 3 = 9 ações (inclui 0 e os extremos).
    """

    def __init__(self, env: gym.Env, tau_u_levels: int | Sequence[float] = 3,
                 tau_r_levels: int | Sequence[float] = 3) -> None:
        super().__init__(env)
        low, high = env.action_space.low, env.action_space.high
        self.u_levels = _levels(tau_u_levels, low[0], high[0])
        self.r_levels = _levels(tau_r_levels, low[1], high[1])
        self.table = np.array(list(itertools.product(self.u_levels, self.r_levels)), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.table))

    def action(self, action) -> np.ndarray:
        return self.table[int(action)]


class MultiDiscreteActionWrapper(gym.ActionWrapper):
    """Ação ``MultiDiscrete([n_u, n_r])``: um índice por componente (tau_u, tau_r)."""

    def __init__(self, env: gym.Env, tau_u_levels: int | Sequence[float] = 3,
                 tau_r_levels: int | Sequence[float] = 3) -> None:
        super().__init__(env)
        low, high = env.action_space.low, env.action_space.high
        self.u_levels = _levels(tau_u_levels, low[0], high[0])
        self.r_levels = _levels(tau_r_levels, low[1], high[1])
        self.action_space = spaces.MultiDiscrete([len(self.u_levels), len(self.r_levels)])

    def action(self, action) -> np.ndarray:
        i, j = np.asarray(action, dtype=np.int64)
        return np.array([self.u_levels[i], self.r_levels[j]], dtype=np.float32)


ACTION_WRAPPERS = {
    "discrete": DiscreteActionWrapper,
    "multidiscrete": MultiDiscreteActionWrapper,
}
