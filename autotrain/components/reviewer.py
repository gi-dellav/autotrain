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
        rewrite_mode: bool = False,
    ):
        self.model = model
        self.prompt = prompt
        self.inference_config = inference_config
        self._experts = experts or []
        self.rewrite_mode = rewrite_mode

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

    def verify(self, samples: list["Sample"], executor=None) -> list["Sample"]:
        """
        Verify the correctness of samples in parallel.

        Args:
            samples: Samples to verify
            executor: Optional shared ThreadPoolExecutor to use

        Returns:
            Verified samples (only correct or rewritten ones)
        """
        from concurrent.futures import ThreadPoolExecutor

        if not self._experts:
            # No expert, pass all samples
            for sample in samples:
                sample.metadata["check"] = {
                    "is_correct": True,
                    "explanation": "No checker available",
                    "checker": "none",
                }
            return samples

        def _verify_one(sample):
            expert = self._experts[0]
            check_result = expert.check(
                sample.input_data, sample.output_data, metadata=sample.metadata
            )
            sample.metadata["check"] = check_result

            is_correct = check_result.get("is_correct", False)
            skipped = check_result.get("skipped", False)

            if not is_correct and not skipped and self.rewrite_mode:
                explanation = check_result.get("explanation", "")
                sample.metadata["original_output"] = sample.output_data
                rewritten_output = expert.rewrite(
                    sample.input_data, sample.output_data, feedback=explanation
                )
                sample.output_data = rewritten_output
                sample.metadata["rewritten"] = True
                return sample
            elif is_correct or skipped:
                return sample
            return None

        max_workers = self.model.inference_config.max_workers
        if not isinstance(max_workers, int):
            max_workers = 8

        if executor is not None:
            # Use shared executor
            results = list(executor.map(_verify_one, samples))
        else:
            # Create own executor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_verify_one, samples))

        return [r for r in results if r is not None]

    async def verify_async(self, samples: list["Sample"], executor=None) -> list["Sample"]:
        """
        Verify the correctness of samples in parallel (async).

        Args:
            samples: Samples to verify
            executor: AsyncExecutor to use (required for async)

        Returns:
            Verified samples (only correct or rewritten ones)
        """
        from concurrent.futures import (
            ThreadPoolExecutor,
        )  # not used but keep import? Actually not needed

        if not self._experts:
            # No expert, pass all samples
            for sample in samples:
                sample.metadata["check"] = {
                    "is_correct": True,
                    "explanation": "No checker available",
                    "checker": "none",
                }
            return samples

        async def _verify_one(sample):
            expert = self._experts[0]
            check_result = await expert.check_async(
                sample.input_data, sample.output_data, metadata=sample.metadata
            )
            sample.metadata["check"] = check_result

            is_correct = check_result.get("is_correct", False)
            skipped = check_result.get("skipped", False)

            if not is_correct and not skipped and self.rewrite_mode:
                explanation = check_result.get("explanation", "")
                sample.metadata["original_output"] = sample.output_data
                rewritten_output = await expert.rewrite_async(
                    sample.input_data, sample.output_data, feedback=explanation
                )
                sample.output_data = rewritten_output
                sample.metadata["rewritten"] = True
                return sample
            elif is_correct or skipped:
                return sample
            return None

        if executor is None:
            raise ValueError("executor is required for async verify")

        results = await executor.map_async(_verify_one, samples)
        return [r for r in results if r is not None]


__all__ = ["Checker"]
