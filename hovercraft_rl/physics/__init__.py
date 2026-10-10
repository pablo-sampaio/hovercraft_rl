"""Modelos físicos (simuladores) intercambiáveis."""

from __future__ import annotations

from .base import PhysicsModel
from .hovercraft3dof import Hovercraft3DOF, Hovercraft3DOFParams
from .hovercraft3dof_drag import Hovercraft3DOFDrag, Hovercraft3DOFDragParams

# Registro por nome: adicione aqui novos modelos para trocá-los via string.
PHYSICS_MODELS: dict[str, type[PhysicsModel]] = {
    "hovercraft3dof": Hovercraft3DOF,
    "hovercraft3dof_drag": Hovercraft3DOFDrag,
}


def make_physics(spec: str | PhysicsModel | None = None, params=None) -> PhysicsModel:
    """Cria um modelo a partir do nome registrado, ou devolve a instância recebida."""
    if isinstance(spec, PhysicsModel):
        return spec
    name = spec or "hovercraft3dof"
    try:
        cls = PHYSICS_MODELS[name]
    except KeyError:
        raise KeyError(f"Modelo físico desconhecido: {name!r}. Disponíveis: {list(PHYSICS_MODELS)}")
    return cls(params) if params is not None else cls()


__all__ = ["PhysicsModel", "Hovercraft3DOF", "Hovercraft3DOFParams", "Hovercraft3DOFDrag",
           "Hovercraft3DOFDragParams", "PHYSICS_MODELS", "make_physics"]
