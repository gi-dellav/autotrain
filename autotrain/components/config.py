"""Component configuration classes."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autotrain.config import InferenceConfig
    from autotrain.expert import Expert


@dataclass
class ComponentConfig:
    """Base configuration for components."""

    prompt: str
    inference_config: "InferenceConfig"


@dataclass
class ExpertWeight:
    """Weight configuration for an expert in multi-expert setup."""

    expert: "Expert"
    weight: float  # Production rate weight (will be normalized)

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Expert weight must be non-negative")


__all__ = ["ComponentConfig", "ExpertWeight"]
