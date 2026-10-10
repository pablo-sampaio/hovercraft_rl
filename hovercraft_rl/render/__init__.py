"""Visualização (pygame). Um renderizador por modelo físico."""

from __future__ import annotations

from ..physics import Hovercraft3DOF, PhysicsModel
from .base import BaseTrackRenderer
from .camera import VIEW_LABELS, Camera
from .hovercraft_renderer import HovercraftRenderer

# Associa cada modelo físico ao seu renderizador. Modelos novos: registre aqui.
RENDERERS: dict[type[PhysicsModel], type[BaseTrackRenderer]] = {
    Hovercraft3DOF: HovercraftRenderer,
}


def make_renderer(physics: PhysicsModel, track, **kwargs) -> BaseTrackRenderer:
    """Cria o renderizador adequado ao modelo físico (ou o genérico, se não houver)."""
    for cls in type(physics).__mro__:
        if cls in RENDERERS:
            rcls = RENDERERS[cls]
            break
    else:
        rcls = BaseTrackRenderer
    if issubclass(rcls, HovercraftRenderer):
        kwargs.setdefault("action_low", physics.action_low)
        kwargs.setdefault("action_high", physics.action_high)
    return rcls(track, **kwargs)


__all__ = ["BaseTrackRenderer", "HovercraftRenderer", "Camera", "VIEW_LABELS", "RENDERERS", "make_renderer"]
