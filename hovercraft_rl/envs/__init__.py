"""Ambiente Gymnasium, observações e recompensas."""

from .observations import OBSERVATIONS, BodyFrameObservation, ObservationBuilder, ReferenceErrorObservation
from .rewards import (CompositeReward, ControlEffortPenalty, LateralErrorPenalty, OffTrackPenalty,
                      ProgressReward, RewardTerm, default_reward)
from .track_env import HovercraftTrackEnv, make_env
from .wrappers import ACTION_WRAPPERS, DiscreteActionWrapper, MultiDiscreteActionWrapper

__all__ = [
    "HovercraftTrackEnv", "make_env",
    "DiscreteActionWrapper", "MultiDiscreteActionWrapper", "ACTION_WRAPPERS",
    "ObservationBuilder", "ReferenceErrorObservation", "BodyFrameObservation", "OBSERVATIONS",
    "RewardTerm", "CompositeReward", "ProgressReward", "OffTrackPenalty",
    "LateralErrorPenalty", "ControlEffortPenalty", "default_reward",
]
