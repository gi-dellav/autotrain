"""Checker component for verifying solution correctness."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from autotrain.config import InferenceConfig
    from autotrain.core.model import Model
    from autotrain.data_types import Sample
    from autotrain.expert import Expert


class Checker:
    """
    Checker component - verifies solution correctness.

    Optional component that can be enabled/disabled.
    """

    def __init__(
        self,
        model: "Model",
        prompt: Union[str, list[str]],
        inference_config: "InferenceConfig",
        experts: Optional[list["Expert"]] = None,
    ):
        self.model = model
        self.prompt = prompt
        self.inference_config = inference_config
        self._experts = experts or []

    @property
    def experts(self) -> list["Expert"]:
        """Get list of experts."""
        return self._experts.copy()

    def add_expert(self, expert: "Expert") -> None:
        """Add an expert for checking."""
        if expert not in self._experts:
            self._experts.append(expert)

    def remove_expert(self, expert: "Expert") -> None:
        """Remove an expert."""
        if expert in self._experts:
            self._experts.remove(expert)

    def verify(self, samples: list["Sample"]) -> list["Sample"]:
        """
        Verify the correctness of samples.

        Args:
            samples: Samples to verify

        Returns:
            Verified samples (only correct ones)
        """
        verified = []

        for sample in samples:
            if self._experts:
                # Use first expert for checking
                expert = self._experts[0]
                check_result = expert.check(
                    sample.input_data, sample.output_data, metadata=sample.metadata
                )
                sample.metadata["check"] = check_result

                # Only keep correct samples (or skipped samples if avoid_checking_same_sample is True)
                if check_result.get("is_correct", False) or check_result.get("skipped"):
                    verified.append(sample)
            else:
                # No expert, pass all samples
                sample.metadata["check"] = {
                    "is_correct": True,
                    "explanation": "No checker available",
                    "checker": "none",
                }
                verified.append(sample)

        return verified


__all__ = ["Checker"]
