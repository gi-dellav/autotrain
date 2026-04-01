"""Component configuration classes."""

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from autotrain.config import InferenceConfig
    from autotrain.expert import Expert


def _get_prompt(prompt: Optional[Union[str, list[str]]]) -> Optional[str]:
    """Get a single prompt from str or list[str], selecting randomly if list."""
    if prompt is None:
        return None
    if isinstance(prompt, list):
        if not prompt:
            return None
        return random.choice(prompt)
    return prompt


@dataclass
class ComponentConfig:
    """Base configuration for components."""

    prompt: Union[str, list[str]]
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
