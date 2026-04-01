"""Splitter component - selects useful samples."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from autotrain.config import InferenceConfig
    from autotrain.core.model import Model
    from autotrain.data_types import Sample
    from autotrain.expert import Expert


class Splitter:
    """
    Splitter component - selects useful samples.

    Groups samples based on sample_multiplier and selects the most useful one.
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
        """Add an expert for selection."""
        if expert not in self._experts:
            self._experts.append(expert)

    def remove_expert(self, expert: "Expert") -> None:
        """Remove an expert."""
        if expert in self._experts:
            self._experts.remove(expert)

    def select(self, samples: list["Sample"], target_count: int) -> list["Sample"]:
        """
        Select useful samples from the input.

        Args:
            samples: List of samples to select from
            target_count: Target number of samples to return

        Returns:
            Selected samples
        """
        if len(samples) <= target_count:
            return samples

        # Group samples and select the best from each group
        sample_multiplier = self.model.sample_multiplier
        selected = []

        for i in range(0, len(samples), sample_multiplier):
            group = samples[i : i + sample_multiplier]

            if len(group) == 1:
                selected.append(group[0])
            else:
                best = self._select_best(group)
                if best is not None:
                    selected.append(best)

        # Ensure we return exactly target_count samples
        return selected[:target_count]

    def _select_best(self, samples: list["Sample"]) -> Optional["Sample"]:
        """
        Select the best sample from a group.

        Uses expert comparison when available, otherwise falls back to
        heuristic-based selection (length, diversity, etc.).

        Args:
            samples: List of samples to select from

        Returns:
            Selected best sample
        """
        if len(samples) < 2:
            return samples[0] if samples else None

        # Try to use expert for selection if available
        if self._experts:
            try:
                expert = self._experts[0]
                sample_outputs = [s.output_data for s in samples]

                # Use expert's compare method for pairwise comparison
                if len(sample_outputs) == 2:
                    comparison = expert.compare(
                        input_data=samples[0].input_data,
                        output_a=sample_outputs[0],
                        output_b=sample_outputs[1],
                    )
                    winner_idx = 0 if comparison["winner"] == "a" else 1
                    if comparison["winner"] == "tie":
                        winner_idx = 0  # Default to first on tie

                    selected = samples[winner_idx]
                    selected.metadata["selected_by"] = f"expert:{expert.model_name}"
                    selected.metadata["comparison_result"] = comparison
                    return selected
                else:
                    # Multiple samples: use expert's select method
                    selected_output = expert.select(sample_outputs)
                    for sample in samples:
                        if sample.output_data == selected_output:
                            sample.metadata["selected_by"] = f"expert:{expert.model_name}"
                            return sample
            except Exception as e:
                print(f"Warning: Expert selection failed, using fallback: {e}")

        # Fallback: Heuristic-based selection
        # Score samples based on multiple criteria
        scored_samples = []
        for sample in samples:
            score = self._score_sample(sample)
            scored_samples.append((sample, score))

        # Select highest scored sample
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        best_sample = scored_samples[0][0]
        best_sample.metadata["selected_by"] = "default"  # Keep "default" for backward compatibility
        best_sample.metadata["selection_score"] = scored_samples[0][1]
        return best_sample

    def _score_sample(self, sample: "Sample") -> float:
        """
        Score a sample based on quality heuristics.

        Args:
            sample: Sample to score

        Returns:
            Quality score (higher = better)
        """
        output = sample.output_data
        score = 0.0

        # Length score (prefer moderate length, not too short or too long)
        length = len(output.split())
        if 20 <= length <= 200:
            score += 3.0
        elif 10 <= length < 20 or 200 < length <= 500:
            score += 2.0
        elif length > 500:
            score += 1.0  # Too long
        else:
            score += 0.5  # Too short

        # Completeness score (check for common completion indicators)
        if output and output[-1] in ".!?":
            score += 1.0  # Properly ended

        # Diversity score (check for varied vocabulary)
        words = output.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 2.0  # More diverse = better

        # Coherence indicators
        coherence_words = [
            "therefore",
            "because",
            "however",
            "thus",
            "hence",
            "first",
            "second",
            "finally",
            "in conclusion",
            "for example",
        ]
        for word in coherence_words:
            if word in output.lower():
                score += 0.5
                break

        return score


__all__ = ["Splitter"]
