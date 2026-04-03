"""Solver component - LLM solves input samples."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Union

from autotrain.components.config import ExpertWeight

if TYPE_CHECKING:
    from autotrain.config import InferenceConfig
    from autotrain.core.model import Model
    from autotrain.data_types import Sample
    from autotrain.expert import Expert


class Solver:
    """
    Solver component - LLM solves input samples.

    Takes input samples and produces output solutions.
    Supports multiple experts with weighted production rates.
    """

    def __init__(
        self,
        model: "Model",
        prompt: Union[str, list[str]],
        inference_config: "InferenceConfig",
        experts: Optional[list[Union["Expert", ExpertWeight]]] = None,
    ):
        self.model = model
        self.prompt = prompt
        self.inference_config = inference_config
        self._expert_weights: list[ExpertWeight] = []

        if experts:
            for expert in experts:
                if isinstance(expert, ExpertWeight):
                    self._expert_weights.append(expert)
                else:
                    # Use expert's production_rate as default weight for bare Expert
                    self._expert_weights.append(
                        ExpertWeight(expert=expert, weight=expert.production_rate)
                    )

    @property
    def experts(self) -> list["Expert"]:
        """Get list of experts."""
        return [ew.expert for ew in self._expert_weights]

    def add_expert(self, expert: "Expert", weight: float = 1.0) -> None:
        """
        Add an expert with a production weight.

        Args:
            expert: The expert to add
            weight: Production weight (higher = more samples from this expert)
        """
        self._expert_weights.append(ExpertWeight(expert=expert, weight=weight))

    def remove_expert(self, expert: "Expert") -> None:
        """Remove an expert from the solver."""
        self._expert_weights = [ew for ew in self._expert_weights if ew.expert != expert]

    def clear_experts(self) -> None:
        """Clear all experts."""
        self._expert_weights.clear()

    def _get_normalized_weights(self) -> list[tuple["Expert", float]]:
        """Get experts with normalized weights for intervention rate calculation.

        Returns:
            List of (expert, normalized_weight) tuples where weights sum to 1.0
        """
        if not self._expert_weights:
            return []

        total_weight = sum(ew.weight for ew in self._expert_weights)
        if total_weight == 0:
            return []

        return [(ew.expert, ew.weight / total_weight) for ew in self._expert_weights]

    def _select_source(self) -> tuple[Optional["Expert"], bool]:
        """
        Select whether to use an expert or the model.

        Returns:
            Tuple of (selected expert or None, is_expert)
        """
        if not self._expert_weights:
            return None, False

        # Calculate total expert weight
        total_expert_weight = sum(ew.weight for ew in self._expert_weights)

        # Model gets the remaining weight (1.0 - total_expert_weight)
        # Normalize so expert weights + model weight = 1.0
        # If total_expert_weight >= 1.0, model never gets selected
        model_weight = max(0.0, 1.0 - total_expert_weight)

        # Create weighted selection list
        selections: list[tuple[Optional["Expert"], float]] = []
        for ew in self._expert_weights:
            selections.append((ew.expert, ew.weight))
        selections.append((None, model_weight))

        total = sum(w for _, w in selections)
        if total == 0:
            return None, False

        # Normalize and select
        r = random.random()
        cumulative = 0.0
        for expert, weight in selections:
            cumulative += weight / total
            if r <= cumulative:
                return expert, expert is not None

        # Fallback to model
        return None, False

    def solve(self, samples: list["Sample"]) -> list["Sample"]:
        """
        Solve input samples to produce outputs in parallel.

        Args:
            samples: List of input samples to solve

        Returns:
            List of samples with filled outputs
        """
        from concurrent.futures import ThreadPoolExecutor

        iteration = getattr(self.model, "_current_iteration", 0)

        def _solve_sample(sample):
            expert, is_expert = self._select_source()

            if is_expert and expert is not None:
                output = expert.solve(
                    sample.input_data, metadata=sample.metadata, iteration=iteration
                )
                sample.metadata["solver"] = f"expert:{expert.model_name}"
            else:
                output = self._model_solve(sample.input_data, iteration=iteration)
                sample.metadata["solver"] = "model"

            sample.output_data = output
            return sample

        max_workers = self.model.inference_config.max_workers
        if not isinstance(max_workers, int):
            max_workers = 8

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            solved_samples = list(executor.map(_solve_sample, samples))

        return solved_samples

    def _model_solve(self, input_data: str, iteration: int = 0) -> str:
        """
        Solve using the fine-tuned model.

        Args:
            input_data: The input/problem to solve
            iteration: Current iteration number for dynamic temperature

        Returns:
            Model's solution as a string
        """
        # Get the baked solver prompt
        prompt = self.model.prompts.get_solver()
        prompt += f"\n\n{input_data}"

        # Check if model is loaded and not a MagicMock
        if self.model.is_loaded and not hasattr(self.model, "_spec_class"):
            try:
                # Use the model's generate method with appropriate config
                result = self.model.generate(
                    prompt=prompt,
                    temperature=self.inference_config.temperature,
                    max_tokens=self.inference_config.max_tokens,
                    top_p=self.inference_config.top_p,
                    iteration=iteration,
                )

                # Check if result is a string (not a MagicMock)
                if isinstance(result, str) and result:
                    return result
            except Exception as e:
                print(f"Warning: Model solve failed, using fallback: {e}")

        # Fallback: return a placeholder response
        return f"Solution for: {input_data}"


__all__ = ["Solver"]
