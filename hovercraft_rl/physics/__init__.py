"""Modelos físicos (simuladores) intercambiáveis."""

from __future__ import annotations

from .base import PhysicsModel
from .hovercraft3dof_drag import Hovercraft3DOFDrag, Hovercraft3DOFParams

# Registro por nome: adicione aqui novos modelos para trocá-los via string.
PHYSICS_MODELS: dict[str, type[PhysicsModel]] = {
    "3dof+drag": Hovercraft3DOFDrag,
}


def make_physics(spec: str | PhysicsModel | None = None, params=None) -> PhysicsModel:
    """Cria um modelo a partir do nome registrado, ou devolve a instância recebida."""
    if isinstance(spec, PhysicsModel):
        return spec
    name = spec or "3dof+drag"
    try:
        cls = PHYSICS_MODELS[name]
    except KeyError:
        raise KeyError(f"Modelo físico desconhecido: {name!r}. Disponíveis: {list(PHYSICS_MODELS)}")
    return cls(params) if params is not None else cls()


__all__ = ["PhysicsModel", "Hovercraft3DOFDrag", "Hovercraft3DOFParams",
           "PHYSICS_MODELS", "make_physics"]
