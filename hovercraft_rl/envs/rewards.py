"""Funções de recompensa modulares: termos independentes combinados com pesos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np


class RewardTerm(ABC):
    """Um termo de recompensa. ``info`` é o dicionário do passo (ver ``envs/README.md``)."""

    name: str = "term"

    def reset(self, env, info: Mapping[str, Any]) -> None:
        """Chamado em ``env.reset`` (para termos com memória)."""

    @abstractmethod
    def __call__(self, env, prev_info: Mapping[str, Any], info: Mapping[str, Any],
                 action: np.ndarray, terminated: bool) -> float: ...


class ProgressReward(RewardTerm):
    """Avanço ao longo da linha base no passo (m): ``scale * Δs``. Negativo se andar para trás."""

    name = "progress"

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(self, env, prev_info, info, action, terminated) -> float:
        return self.scale * info["progress_delta"]


class OffTrackPenalty(RewardTerm):
    """``-penalty`` no passo em que o veículo sai da pista (fim do episódio)."""

    name = "off_track"

    def __init__(self, penalty: float = 10.0) -> None:
        self.penalty = penalty

    def __call__(self, env, prev_info, info, action, terminated) -> float:
        return -self.penalty if info["off_track"] else 0.0


class LateralErrorPenalty(RewardTerm):
    """``-(e_lat / half_width)²`` (opcional; desligado por padrão)."""

    name = "lateral_error"

    def __call__(self, env, prev_info, info, action, terminated) -> float:
        return -(info["lateral_error"] / env.track.half_width) ** 2


class ControlEffortPenalty(RewardTerm):
    """``-Σ w_i a_i²`` (opcional; desligado por padrão)."""

    name = "control_effort"

    def __init__(self, weights: Sequence[float] = (0.0, 1.0)) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)

    def __call__(self, env, prev_info, info, action, terminated) -> float:
        return -float(np.sum(self.weights * np.asarray(action, dtype=np.float64) ** 2))


class CompositeReward(RewardTerm):
    """Soma ponderada de termos. Os valores de cada termo vão para ``info["reward_terms"]``."""

    name = "composite"

    def __init__(self, terms: Sequence[tuple[RewardTerm, float]]) -> None:
        self.terms = list(terms)
        self.last_terms: dict[str, float] = {}

    def reset(self, env, info) -> None:
        for t, _ in self.terms:
            t.reset(env, info)
        self.last_terms = {}

    def __call__(self, env, prev_info, info, action, terminated) -> float:
        total = 0.0
        self.last_terms = {}
        for t, w in self.terms:
            val = w * float(t(env, prev_info, info, action, terminated))
            self.last_terms[t.name] = self.last_terms.get(t.name, 0.0) + val
            total += val
        return total


def default_reward(progress_scale: float = 1.0, off_track_penalty: float = 10.0) -> CompositeReward:
    """Recompensa padrão: progresso ao longo da pista + penalidade por sair dela."""
    return CompositeReward([(ProgressReward(progress_scale), 1.0), (OffTrackPenalty(off_track_penalty), 1.0)])
