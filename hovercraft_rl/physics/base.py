"""Interface comum para modelos físicos.

Um modelo físico só conhece seu próprio estado e as ações. Ele não sabe nada
sobre pista, recompensa, sucesso ou falha.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np


class PhysicsModel(ABC):
    """Classe base de um simulador físico discreto.

    Subclasses devem definir ``state_names``, ``action_names``, ``action_low``,
    ``action_high`` e ``dt``, e implementar ``derivatives`` e/ou ``step``.
    """

    state_names: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._state = np.zeros(len(self.state_names), dtype=np.float64)
        self.t = 0.0

    # --- propriedades obrigatórias -------------------------------------------------
    @property
    @abstractmethod
    def dt(self) -> float:
        """Tempo de amostragem (s)."""

    @property
    @abstractmethod
    def action_low(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def action_high(self) -> np.ndarray: ...

    @abstractmethod
    def step(self, action: Sequence[float]) -> np.ndarray:
        """Avança um passo ``dt`` aplicando ``action``; retorna o novo estado."""

    # --- utilidades comuns -------------------------------------------------------------
    @property
    def state(self) -> np.ndarray:
        """Cópia do vetor de estado atual."""
        return self._state.copy()

    def state_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.state_names, self._state)}

    def reset(self, state: Mapping[str, float] | Sequence[float] | None = None) -> np.ndarray:
        """Reinicia o tempo e define o estado (dict parcial, vetor completo ou zeros)."""
        new = np.zeros(len(self.state_names), dtype=np.float64)
        if isinstance(state, Mapping):
            unknown = set(state) - set(self.state_names)
            if unknown:
                raise KeyError(f"Variáveis de estado desconhecidas: {sorted(unknown)}")
            for i, name in enumerate(self.state_names):
                new[i] = float(state.get(name, 0.0))
        elif state is not None:
            arr = np.asarray(state, dtype=np.float64)
            if arr.shape != new.shape:
                raise ValueError(f"Estado deve ter forma {new.shape}, recebeu {arr.shape}")
            new = arr.copy()
        self._state = new
        self.t = 0.0
        return self.state

    def clip_action(self, action: Sequence[float]) -> np.ndarray:
        """Saturação do atuador nos limites do modelo."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != len(self.action_names):
            raise ValueError(f"Ação deve ter {len(self.action_names)} componentes, recebeu {a.shape[0]}")
        return np.clip(a, self.action_low, self.action_high)
